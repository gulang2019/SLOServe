from typing import Dict
from .operation import OpCode, Operator
from .inference_op import CausalLMInferenceOp
from .tokenizer_op import TokenizerDecodeOp, TokenizerEncodeOp
from .delete_op import DeleteOp 
from .concat import ConcatOp
from .verify import VerifyOp
from .get import GetOp

OP_CLASSES: Dict[OpCode, Operator] = {
    OpCode.CausalLMInference: CausalLMInferenceOp,
    OpCode.ENCODE: TokenizerEncodeOp,
    OpCode.DECODE: TokenizerDecodeOp,
    OpCode.DELETE: DeleteOp,
    OpCode.CONCAT: ConcatOp,
    OpCode.VERIFY: VerifyOp,
    OpCode.GET: GetOp 
}
