#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "loop-unroll-limited"

STATISTIC(NumUnrolled, "Number of loops unrolled");
STATISTIC(NumUnrollFailed, "Number of loops that could not be unrolled");

namespace {

class LoopUnrollPass : public MachineFunctionPass {
public:
  static char ID;
  LoopUnrollPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  bool unrollLoop(MachineLoop *L, MachineFunction &MF, MachineLoopInfo &MLI);
  bool canUnrollLoop(MachineLoop *L, int &TripCount, int &UnrollFactor);
  int getTripCount(MachineLoop *L);
  int computeUnrollFactor(int TripCount);
  bool performUnrolling(MachineLoop *L, MachineFunction &MF, int UnrollFactor,
                        MachineLoopInfo &MLI);
  void cloneLoopBody(MachineLoop *L, MachineFunction &MF, int Copies,
                     MachineBasicBlock *InsertBefore, MachineLoopInfo &MLI);
  void adjustInductionVariable(MachineLoop *L, int Factor);
  MachineInstr *findInductionIncrement(MachineBasicBlock *Latch);
  void collectLoops(MachineLoop *L, SmallVectorImpl<MachineLoop *> &Loops);
};

char LoopUnrollPass::ID = 0;

}

static RegisterPass<LoopUnrollPass> X("loop-unroll-limited",
                                      "Loop unroll pass (max 5 iterations)",
                                      false, false);

bool LoopUnrollPass::runOnMachineFunction(MachineFunction &MF) {
  auto &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();

  SmallVector<MachineLoop *, 8> AllLoops;

  for (MachineLoop *L : MLI)
    collectLoops(L, AllLoops);

  bool Changed = false;
  for (MachineLoop *L : AllLoops) {
    if (unrollLoop(L, MF, MLI))
      Changed = true;
  }

  return Changed;
}

void LoopUnrollPass::collectLoops(MachineLoop *L,
                                  SmallVectorImpl<MachineLoop *> &Loops) {
  for (MachineLoop *SubLoop : *L)
    collectLoops(SubLoop, Loops);
  Loops.push_back(L);
}

bool LoopUnrollPass::unrollLoop(MachineLoop *L, MachineFunction &MF,
                                MachineLoopInfo &MLI) {
  int TripCount = 0;
  int UnrollFactor = 0;

  if (!canUnrollLoop(L, TripCount, UnrollFactor)) {
    NumUnrollFailed++;
    return false;
  }

  LLVM_DEBUG(dbgs() << "Unrolling loop (TC=" << TripCount
                    << ", Factor=" << UnrollFactor << ")\n");

  bool Result = performUnrolling(L, MF, UnrollFactor, MLI);
  if (Result)
    NumUnrolled++;

  return Result;
}

bool LoopUnrollPass::canUnrollLoop(MachineLoop *L, int &TripCount,
                                   int &UnrollFactor) {
  if (!L->getLoopPreheader()) {
    LLVM_DEBUG(dbgs() << "Loop has no preheader\n");
    return false;
  }

  if (!L->getLoopLatch()) {
    LLVM_DEBUG(dbgs() << "Loop has no single latch\n");
    return false;
  }

  TripCount = getTripCount(L);
  if (TripCount <= 1) {
    LLVM_DEBUG(dbgs() << "Unknown or trivial trip count: " << TripCount
                      << "\n");
    return false;
  }

  UnrollFactor = computeUnrollFactor(TripCount);
  if (UnrollFactor <= 1) {
    LLVM_DEBUG(dbgs() << "No suitable unroll factor for TC=" << TripCount
                      << "\n");
    return false;
  }

  int LoopBlocks = L->getNumBlocks();
  if (LoopBlocks > 10 && UnrollFactor > 2) {
    LLVM_DEBUG(dbgs() << "Loop too large (" << LoopBlocks << " blocks)\n");
    return false;
  }

  return true;
}

int LoopUnrollPass::getTripCount(MachineLoop *L) {
  MachineBasicBlock *Latch = L->getLoopLatch();
  if (!Latch)
    return -1;

  for (auto &MI : reverse(*Latch)) {
    if (MI.isBranch() && MI.getNumOperands() >= 2) {
      for (const MachineOperand &Op : MI.operands()) {
        if (Op.isImm() && Op.getImm() > 0) {
          return (int)Op.getImm();
        }
      }
    }

    if (MI.getOpcode() == 151) {
      for (const MachineOperand &Op : MI.operands()) {
        if (Op.isImm() && Op.getImm() > 0)
          return (int)Op.getImm();
      }
    }
  }

  return -1;
}

int LoopUnrollPass::computeUnrollFactor(int TripCount) {
  const int MAX_UNROLL = 5;

  if (TripCount <= MAX_UNROLL)
    return TripCount;

  for (int Factor = MAX_UNROLL; Factor >= 2; --Factor) {
    if (TripCount % Factor == 0)
      return Factor;
  }

  return 1;
}

bool LoopUnrollPass::performUnrolling(MachineLoop *L, MachineFunction &MF,
                                      int UnrollFactor, MachineLoopInfo &MLI) {
  MachineBasicBlock *Preheader = L->getLoopPreheader();
  MachineBasicBlock *Latch = L->getLoopLatch();
  MachineBasicBlock *Exit = L->getExitBlock();

  if (!Preheader || !Latch || !Exit)
    return false;

  SmallVector<MachineBasicBlock *, 8> LoopBlocks(L->block_begin(),
                                                 L->block_end());

  int Copies = UnrollFactor - 1;

  if (Copies > 0) {
    cloneLoopBody(L, MF, Copies, Latch, MLI);
  }

  adjustInductionVariable(L, UnrollFactor);

  if (Preheader->getSuccessors().size() > 0) {
    Preheader->ReplaceSuccessorWith(Latch, Exit);
  }

  return true;
}

void LoopUnrollPass::cloneLoopBody(MachineLoop *L, MachineFunction &MF,
                                   int Copies, MachineBasicBlock *InsertBefore,
                                   MachineLoopInfo &MLI) {
  SmallVector<MachineBasicBlock *, 8> LoopBlocks(L->block_begin(),
                                                 L->block_end());

  for (int i = 0; i < Copies; ++i) {
    for (MachineBasicBlock *BB : LoopBlocks) {
      if (BB == L->getLoopPreheader() || BB == L->getExitBlock())
        continue;

      MachineBasicBlock *CloneBB = MF.CreateMachineBasicBlock();

      for (MachineInstr &MI : *BB) {
        MachineInstr *CloneMI = MF.CloneMachineInstr(&MI);
        CloneBB->insert(CloneBB->end(), CloneMI);
      }

      MF.insert(InsertBefore->getIterator(), CloneBB);
      CloneBB->cloneSuccessors(BB);
    }
  }
}

void LoopUnrollPass::adjustInductionVariable(MachineLoop *L, int Factor) {
  MachineBasicBlock *Latch = L->getLoopLatch();
  if (!Latch)
    return;

  MachineInstr *IncInstr = findInductionIncrement(Latch);
  if (IncInstr) {
    for (MachineOperand &Op : IncInstr->operands()) {
      if (Op.isImm() && Op.getImm() == 1) {
        Op.setImm(Factor);
        LLVM_DEBUG(dbgs() << "Updated induction step to " << Factor << "\n");
        break;
      }
    }
  }
}

MachineInstr *LoopUnrollPass::findInductionIncrement(MachineBasicBlock *Latch) {
  for (MachineInstr &MI : *Latch) {
    if (MI.getOpcode() == 0x04 || MI.getOpcode() == 0x81) {
      for (const MachineOperand &Op : MI.operands()) {
        if (Op.isImm() && Op.getImm() == 1)
          return &MI;
      }
    }
  }
  return nullptr;
}

namespace llvm {
FunctionPass *createLoopUnrollLimitedPass() { return new LoopUnrollPass(); }
}