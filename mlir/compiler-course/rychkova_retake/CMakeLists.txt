get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(TARGET_NAME "${CURRENT_DIR_NAME}_MLIR")

file(GLOB_RECURSE SOURCES *.cpp *.h *.hpp)

add_llvm_pass_plugin(${TARGET_NAME}
    ${SOURCES}
    DEPENDS
    intrinsics_gen
    MLIRBuiltinLocationAttributesIncGen

    LINK_LIBS
    MLIRArithDialect
    MLIRMathDialect
    MLIRFuncDialect
    MLIRIR
    MLIRPass
    MLIRRewrite
    MLIRSupport
    MLIRTransformUtils

    BUILDTREE_ONLY
)

set(MLIR_TEST_DEPENDS ${TARGET_NAME} ${MLIR_TEST_DEPENDS} PARENT_SCOPE)
