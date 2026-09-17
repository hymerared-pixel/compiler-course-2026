// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/rychkova_retake_MLIR%shlibext --pass-pipeline="builtin.module(func.func(fma-fusion))" %s | FileCheck %s

// CHECK-LABEL: func @simple_fma
func.func @simple_fma(%a: f32, %b: f32, %c: f32) -> f32 {
  // CHECK-NOT: arith.mulf
  // CHECK: %[[R:.*]] = math.fma %arg0, %arg1, %arg2 : f32
  %m = arith.mulf %a, %b : f32
  %r = arith.addf %m, %c : f32
  // CHECK: return %[[R]]
  return %r : f32
}

// CHECK-LABEL: func @commuted_fma
func.func @commuted_fma(%a: f32, %b: f32, %c: f32) -> f32 {
  // CHECK: math.fma %arg0, %arg1, %arg2 : f32
  %m = arith.mulf %a, %b : f32
  %r = arith.addf %c, %m : f32
  return %r : f32
}

// CHECK-LABEL: func @no_fuse_multi_use
func.func @no_fuse_multi_use(%a: f32, %b: f32, %c: f32) -> (f32, f32) {
  // The mul result is used twice, so fusing would duplicate work —
  // pattern should leave this alone.
  // CHECK: arith.mulf
  // CHECK: arith.addf
  %m = arith.mulf %a, %b : f32
  %r = arith.addf %m, %c : f32
  return %r, %m : f32, f32
}
