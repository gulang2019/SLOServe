from typing import List
from dataclasses import dataclass
from typing import Tuple, Callable
import numpy as np

# the log is an loop of batches
'''
...
[1;36m(EngineCore_0 pid=4038994)[0;0m EXECUTE MODEL
[1;36m(EngineCore_0 pid=4038994)[0;0m context_lengths:[282, 115, 224, 56, 88, 385, 45, 1228, 634, 542, 128, 352, 455, 298, 180, 404, 347, 367, 402, 578, 170, 451, 642, 293, 358, 881, 336, 656, 310, 498, 582, 296, 61, 426, 103, 357, 592, 740, 80, 62, 556, 94, 134, 85, 362, 115, 280, 300, 71, 47, 259, 306, 310, 810, 257, 203, 214, 257, 243, 61, 107, 139, 906, 144, 49, 871, 841, 701, 694, 173, 264, 191, 88, 215, 236, 181, 203, 49], current_lengths:[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
[1;36m(EngineCore_0 pid=4038994)[0;0m PROFILING DONE, current_lengths:[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], context_lengths:[282, 115, 224, 56, 88, 385, 45, 1228, 634, 542, 128, 352, 455, 298, 180, 404, 347, 367, 402, 578, 170, 451, 642, 293, 358, 881, 336, 656, 310, 498, 582, 296, 61, 426, 103, 357, 592, 740, 80, 62, 556, 94, 134, 85, 362, 115, 280, 300, 71, 47, 259, 306, 310, 810, 257, 203, 214, 257, 243, 61, 107, 139, 906, 144, 49, 871, 841, 701, 694, 173, 264, 191, 88, 215, 236, 181, 203, 49]
[1;36m(EngineCore_0 pid=4038994)[0;0m Execute[28209]: 0.023401498794555664 seconds
[1;36m(EngineCore_0 pid=4038994)[0;0m Schedule[28210]: SchedulerOutput(
    scheduled_new_reqs=[], 
    scheduled_cached_reqs=CachedRequestData(
        req_ids=[
            'chatcmpl-ca0eb290086846a5ba7db72b37c9af04',
            'chatcmpl-973252b146274a12ad78b68fb8f1854c',
            'chatcmpl-ea15dadc22dd439a912cb6fb97364b3f',
            'chatcmpl-79f00ba6e9a74302b4579a1295484e1a',
            'chatcmpl-804a6497c2f2438b97d8eecc67d537bc',
            'chatcmpl-281629dee711427080798c6381accee4',
            'chatcmpl-10b8f0daffb7495d86742681da16d889',
            'chatcmpl-4cdb3724c693449e975ede6ad228689f',
            'chatcmpl-50084383e33d4faba212d8e41dbe3baf',
            'chatcmpl-fe699ac7715f4b6a8e64144e546b643d',
            'chatcmpl-61fbd069ea544335946f4b595525ace9',
            'chatcmpl-c40052e9b03e461b93447480101d7bc7',
            'chatcmpl-af8d737244d44a888f3b2b5c75f7c1f1',
            'chatcmpl-324b018fca324746868c760fe1ad07fd',
            'chatcmpl-c5f19970e6b5453b9ea3bbc0944813d0',
            'chatcmpl-f7c4c4b56d1d4bb9a6002f6c79bb9eae',
            'chatcmpl-ddb352d9417744f3a72e97d5c01d8698',
            'chatcmpl-5d155f7963d84036ab35ce163ccb537d',
            'chatcmpl-0453b3fc5a30429580127f2bb43cbb30',
            'chatcmpl-0d1da8787d7547d39d2b50bed20be4d1',
            'chatcmpl-4e4b8b92ac1549f19ab67fb328e6da8c',
            'chatcmpl-f53a5085bbba4f2ca22cc4d3e59ba07a',
            'chatcmpl-8af8ea2d071445d88c7c4285c6e44efc',
            'chatcmpl-9123bdbf7de84cd1a414ef5ce8a4ffa2',
            'chatcmpl-d93d4164a34e47ffa040bfc33d6d0586',
            'chatcmpl-5a3417c290de4195b430e9f1c2a42ee2',
            'chatcmpl-879d91dc4570438db5d507ad2d452a21',
            'chatcmpl-bf8524db4792492ab07e868c7cc5eac1',
            'chatcmpl-4e1f1fa71870457a888ab52862621bde',
            'chatcmpl-d53963d76a0e404680ed98efb6b05519', 
            'chatcmpl-46e036da02fc44df865d156c5f5be191',
            'chatcmpl-d9359c2714b34934899e3230c32c7057',
            'chatcmpl-94b3bce435194480a61f4505e9fc2260',
            'chatcmpl-428b25d2a69e46b6ae0906b2d78771d2',
            'chatcmpl-c955d9b753b541a4b8bea46b615f36d8',
            'chatcmpl-232e9fd2e2824d569ac9b51ea4635f5e',
            'chatcmpl-188acfe57f4a43c8a09ad4f349db4051',
            'chatcmpl-b50439ec2ca44bf0bc97a7b4e7d024c8',
            'chatcmpl-1d62e5d4ac9e42f2aaa1e0291bace753',
            'chatcmpl-20f947cc08524473982d39a6b116aebf',
            'chatcmpl-97bb87378b064fb3981681a106d7a8da',
            'chatcmpl-25cbb4bc2af14b70b7762f19d39aa3c1',
            'chatcmpl-278e365581dd42bca3f54f5bc3f8b32a',
            'chatcmpl-fc28f1ea69b44f13890885dc1aea50e0',
            'chatcmpl-1e8edc8422c745938f151f5d382360e6',
            'chatcmpl-621f6ac0d4e9445280feed4748997888',
            'chatcmpl-b0c6dad651b140e5b3f4175091f1c0e7',
            'chatcmpl-170bd5207f7d4aa5b4d58df906f449af',
            'chatcmpl-d1ca6e1d9866484aa05a161d891fad4f',
            'chatcmpl-1bdcde98306548efa7c07766ed24ba89',
            'chatcmpl-eb103e8c8316456c8c552962b8ba127d',
            'chatcmpl-1caddf892bf143649fa5480a465ccc26',
            'chatcmpl-5349e13b62564c78b36b11333486ea6c',
            'chatcmpl-ee979d1e58b34ae5b613a47b6d91cf73', 
            'chatcmpl-89c2993781cb4cceaa23b28600939193', 'chatcmpl-b6dfaa73b3534c14a76466b3f1b9b130', 
            'chatcmpl-0eea4d5122f94c8e98607a141a742514', 'chatcmpl-8fa2039f3a544ff2810d097bd39a6ce8', 'chatcmpl-a85dd9db632b4e5099533c772394b5e9', 
            'chatcmpl-e6a3e6dd427243d799b9640604773204', 'chatcmpl-645fcc9935fd4f88905e8e560d19a510', 'chatcmpl-3640e17df375465e9c0df476d05b47fd', 'chatcmpl-4757ecc2378744f9895f9741c6cfaf58', 'chatcmpl-cb6ee0c211534fe9b11c640b834c0994', 'chatcmpl-0532da460abe4798917cfd4c4a9277b3', 'chatcmpl-64fb14b2520e4e6e8ef308c1f457b02c', 'chatcmpl-8454f56f75f040339a80b3ba012c7942', 'chatcmpl-d9a826bc898b4639bce1667e009e013f', 'chatcmpl-46a5968634b54e2b8c9cafa4e85bf2eb', 'chatcmpl-66416afd024743bf9795377c3c6f764a', 'chatcmpl-62a847ca232644739aae2fce7f8bd77f', 'chatcmpl-3f2c54686239486c904c882f925a2d3a', 'chatcmpl-6a1b09ec77f0445489a0ede0fb8ca182', 'chatcmpl-59929c35a2834dc6b7012f3f7e1af1a8', 'chatcmpl-989323c16acb4d8384c164be6ee29a68', 'chatcmpl-df7d5416038d4154b7794a61fefb26f3', 'chatcmpl-76b61792cc9d446bbad0089da8d91f58', 'chatcmpl-0ddb7b9bfa124ba18eb39377afc5b5e1'],
            resumed_from_preemption=[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], new_token_ids=[], 
            new_block_ids=[None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, ([11722],), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, ([11673],), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, ([11816],), None, None, None, None, None], 
            num_computed_tokens=[1229, 811, 907, 882, 872, 842, 695, 741, 543, 702, 579, 657, 635, 643, 557, 593, 583, 456, 499, 386, 427, 353, 311, 452, 405, 348, 359, 363, 358, 368, 403, 337, 265, 283, 297, 301, 294, 307, 281, 311, 260, 299, 258, 225, 215, 192, 216, 204, 181, 116, 129, 182, 244, 174, 171, 258, 108, 237, 140, 89, 86, 104, 81, 89, 72, 62, 116, 145, 63, 62, 135, 50, 48, 95, 204, 46, 50, 57]), 
            num_scheduled_tokens={'chatcmpl-ca0eb290086846a5ba7db72b37c9af04': 1, 'chatcmpl-973252b146274a12ad78b68fb8f1854c': 1, 'chatcmpl-ea15dadc22dd439a912cb6fb97364b3f': 1, 'chatcmpl-79f00ba6e9a74302b4579a1295484e1a': 1, 'chatcmpl-804a6497c2f2438b97d8eecc67d537bc': 1, 'chatcmpl-281629dee711427080798c6381accee4': 1, 'chatcmpl-10b8f0daffb7495d86742681da16d889': 1, 'chatcmpl-4cdb3724c693449e975ede6ad228689f': 1, 'chatcmpl-50084383e33d4faba212d8e41dbe3baf': 1, 'chatcmpl-fe699ac7715f4b6a8e64144e546b643d': 1, 'chatcmpl-61fbd069ea544335946f4b595525ace9': 1, 'chatcmpl-c40052e9b03e461b93447480101d7bc7': 1, 'chatcmpl-af8d737244d44a888f3b2b5c75f7c1f1': 1, 'chatcmpl-324b018fca324746868c760fe1ad07fd': 1, 'chatcmpl-c5f19970e6b5453b9ea3bbc0944813d0': 1, 'chatcmpl-f7c4c4b56d1d4bb9a6002f6c79bb9eae': 1, 'chatcmpl-ddb352d9417744f3a72e97d5c01d8698': 1, 'chatcmpl-5d155f7963d84036ab35ce163ccb537d': 1, 'chatcmpl-0453b3fc5a30429580127f2bb43cbb30': 1, 'chatcmpl-0d1da8787d7547d39d2b50bed20be4d1': 1, 'chatcmpl-4e4b8b92ac1549f19ab67fb328e6da8c': 1, 'chatcmpl-f53a5085bbba4f2ca22cc4d3e59ba07a': 1, 'chatcmpl-8af8ea2d071445d88c7c4285c6e44efc': 1, 'chatcmpl-9123bdbf7de84cd1a414ef5ce8a4ffa2': 1, 'chatcmpl-d93d4164a34e47ffa040bfc33d6d0586': 1, 'chatcmpl-5a3417c290de4195b430e9f1c2a42ee2': 1, 'chatcmpl-879d91dc4570438db5d507ad2d452a21': 1, 'chatcmpl-bf8524db4792492ab07e868c7cc5eac1': 1, 'chatcmpl-4e1f1fa71870457a888ab52862621bde': 1, 'chatcmpl-d53963d76a0e404680ed98efb6b05519': 1, 'chatcmpl-46e036da02fc44df865d156c5f5be191': 1, 'chatcmpl-d9359c2714b34934899e3230c32c7057': 1, 'chatcmpl-94b3bce435194480a61f4505e9fc2260': 1, 'chatcmpl-428b25d2a69e46b6ae0906b2d78771d2': 1, 'chatcmpl-c955d9b753b541a4b8bea46b615f36d8': 1, 'chatcmpl-232e9fd2e2824d569ac9b51ea4635f5e': 1, 'chatcmpl-188acfe57f4a43c8a09ad4f349db4051': 1, 'chatcmpl-b50439ec2ca44bf0bc97a7b4e7d024c8': 1, 'chatcmpl-1d62e5d4ac9e42f2aaa1e0291bace753': 1, 'chatcmpl-20f947cc08524473982d39a6b116aebf': 1, 'chatcmpl-97bb87378b064fb3981681a106d7a8da': 1, 'chatcmpl-25cbb4bc2af14b70b7762f19d39aa3c1': 1, 'chatcmpl-278e365581dd42bca3f54f5bc3f8b32a': 1, 'chatcmpl-fc28f1ea69b44f13890885dc1aea50e0': 1, 'chatcmpl-1e8edc8422c745938f151f5d382360e6': 1, 'chatcmpl-621f6ac0d4e9445280feed4748997888': 1, 'chatcmpl-b0c6dad651b140e5b3f4175091f1c0e7': 1, 'chatcmpl-170bd5207f7d4aa5b4d58df906f449af': 1, 'chatcmpl-d1ca6e1d9866484aa05a161d891fad4f': 1, 'chatcmpl-1bdcde98306548efa7c07766ed24ba89': 1, 'chatcmpl-eb103e8c8316456c8c552962b8ba127d': 1, 'chatcmpl-1caddf892bf143649fa5480a465ccc26': 1, 'chatcmpl-5349e13b62564c78b36b11333486ea6c': 1, 'chatcmpl-ee979d1e58b34ae5b613a47b6d91cf73': 1, 'chatcmpl-89c2993781cb4cceaa23b28600939193': 1, 'chatcmpl-b6dfaa73b3534c14a76466b3f1b9b130': 1, 'chatcmpl-0eea4d5122f94c8e98607a141a742514': 1, 'chatcmpl-8fa2039f3a544ff2810d097bd39a6ce8': 1, 'chatcmpl-a85dd9db632b4e5099533c772394b5e9': 1, 'chatcmpl-e6a3e6dd427243d799b9640604773204': 1, 'chatcmpl-645fcc9935fd4f88905e8e560d19a510': 1, 'chatcmpl-3640e17df375465e9c0df476d05b47fd': 1, 'chatcmpl-4757ecc2378744f9895f9741c6cfaf58': 1, 'chatcmpl-cb6ee0c211534fe9b11c640b834c0994': 1, 'chatcmpl-0532da460abe4798917cfd4c4a9277b3': 1, 'chatcmpl-64fb14b2520e4e6e8ef308c1f457b02c': 1, 'chatcmpl-8454f56f75f040339a80b3ba012c7942': 1, 'chatcmpl-d9a826bc898b4639bce1667e009e013f': 1, 'chatcmpl-46a5968634b54e2b8c9cafa4e85bf2eb': 1, 'chatcmpl-66416afd024743bf9795377c3c6f764a': 1, 'chatcmpl-62a847ca232644739aae2fce7f8bd77f': 1, 'chatcmpl-3f2c54686239486c904c882f925a2d3a': 1, 'chatcmpl-6a1b09ec77f0445489a0ede0fb8ca182': 1, 'chatcmpl-59929c35a2834dc6b7012f3f7e1af1a8': 1, 'chatcmpl-989323c16acb4d8384c164be6ee29a68': 1, 'chatcmpl-df7d5416038d4154b7794a61fefb26f3': 1, 'chatcmpl-76b61792cc9d446bbad0089da8d91f58': 1, 'chatcmpl-0ddb7b9bfa124ba18eb39377afc5b5e1': 1}, total_num_scheduled_tokens=78, scheduled_spec_decode_tokens={}, scheduled_encoder_inputs={}, num_common_prefix_blocks=[0], finished_req_ids=set(), free_encoder_input_ids=[], structured_output_request_ids={}, grammar_bitmask=None, kv_connector_metadata=None)[1;36m(APIServer pid=4038954)[0;0m INFO:     127.0.0.1:51764 - "POST /v1/chat/completions HTTP/1.1" 200 OK


...
'''


import re
import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

# ---------- Data models ----------

@dataclass
class Batch:
    context_lengths: List[int]
    current_lengths: List[int]
    elasped_time: float  # keep user's field name (typo) for compatibility

    @property
    def total_current_length(self):
        return sum(self.current_lengths)

    @property
    def average_context_length(self):
        return sum(self.context_lengths) / len(self.context_lengths)

    @property
    def total_multiply(self):
        return sum((a * b) for a, b in zip(self.context_lengths, self.current_lengths))

    @property
    def total_length(self):
        # NOTE: This is exactly as provided by the user. If you intended
        # sum(context_lengths) + sum(current_lengths), change it here.
        return sum(self.context_lengths) + sum(self.context_lengths)


@dataclass
class CachedRequestData:
    req_ids: List[str]
    resumed_from_preemption: List[bool]
    new_token_ids: Optional[List[int]] = None
    new_block_ids: Optional[List[Any]] = None  # can be None/tuples/etc.
    num_computed_tokens: Optional[List[int]] = None


@dataclass
class SchedulerOutput:
    scheduled_new_reqs: List[str]
    scheduled_cached_reqs: Optional[CachedRequestData]
    new_token_ids: Optional[List[int]]
    new_block_ids: Optional[List[Any]]
    num_computed_tokens: Optional[List[int]]
    num_scheduled_tokens: Dict[str, int]
    total_num_scheduled_tokens: int
    scheduled_spec_decode_tokens: Dict[str, Any]
    scheduled_encoder_inputs: Dict[str, Any]
    num_common_prefix_blocks: List[int]
    finished_req_ids: Set[str]
    free_encoder_input_ids: List[Any]
    structured_output_request_ids: Dict[str, Any]
    grammar_bitmask: Optional[Any]
    kv_connector_metadata: Optional[Any]

    # @property
    # def total_num_scheduled_tokens(self):
    #     return sum(self.num_scheduled_tokens.values())
    
    # @property
    def average_num_scheduled_tokens(self):
        return self.total_num_scheduled_tokens / len(self.num_scheduled_tokens)
    
    # @property
    def max_num_scheduled_tokens(self):
        return max(self.num_scheduled_tokens.values())
    
    # @property
    def min_num_scheduled_tokens(self):
        return min(self.num_scheduled_tokens.values())
    
    # @property
    # def median_num_scheduled_tokens(self):
    #     return median(self.num_scheduled_tokens.values())
    
    # def __str__(self):
    #     try: 
    #         return f"SchedulerOutput(#new_reqs={len(self.scheduled_new_reqs)}, \
    #     #cached_reqs={len(self.scheduled_cached_reqs.req_ids)}, \
    #     #computed_tokens={self.num_computed_tokens}, \
    #     #scheduled_tokens={sum(self.num_scheduled_tokens.values())})"
    #     except Exception as e:
    #         return f"SchedulerOutput(error={e})"

    # def __repr__(self):
    #     return self.__str__()

@dataclass
class Iteration:
    scheduler: SchedulerOutput
    batch: Batch

@dataclass
class Request:
    request_id: str
    timestamps: List[float] = field(default_factory=list)
    iterations: List[Tuple[int, float]] = field(default_factory=list)

# ---------- Utilities ----------

_ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
CTX_RE = re.compile(r"context_lengths:\[([^\]]*)\]")
CUR_RE = re.compile(r"current_lengths:\[([^\]]*)\]")
EXEC_TIME_RE = re.compile(r"Execute\[\d+\]:\s*([0-9]*\.?[0-9]+)\s+seconds")

def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)

def _parse_int_list(bracket_contents: str) -> List[int]:
    # Extract integers robustly (ignores stray non-numeric tokens/tuples)
    return list(map(int, re.findall(r"-?\d+", bracket_contents)))


# ---------- Safe literalization of constructor-like reprs ----------

class _CtorToLiteral(ast.NodeTransformer):
    """Turn SchedulerOutput(...)/CachedRequestData(...) into dicts; set() -> empty set."""
    _ctor_names = {"SchedulerOutput", "CachedRequestData"}

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "set" and not node.args and not node.keywords:
            return ast.Set(elts=[])
        if isinstance(node.func, ast.Name) and node.func.id in self._ctor_names:
            keys, vals = [], []
            for kw in node.keywords:
                if kw.arg is None:
                    raise ValueError("Unsupported **kwargs in constructor")
                keys.append(ast.Constant(kw.arg))
                vals.append(kw.value)
            return ast.Dict(keys=keys, values=vals)
        # Disallow other calls (safety)
        raise ValueError("Unsupported call in repr")

def _literalize_constructor_expr(text: str) -> object:
    expr = ast.parse(text, mode="eval")
    expr = _CtorToLiteral().visit(expr)
    ast.fix_missing_locations(expr)
    return ast.literal_eval(expr)

def _to_cached_request_data(obj: Optional[dict]) -> Optional[CachedRequestData]:
    if obj is None:
        return None
    return CachedRequestData(
        req_ids=list(obj.get("req_ids", [])),
        resumed_from_preemption=list(obj.get("resumed_from_preemption", [])),
        new_token_ids=obj.get("new_token_ids"),
        new_block_ids=obj.get("new_block_ids"),
        num_computed_tokens=obj.get("num_computed_tokens"),
    )

def _to_scheduler_output(obj: dict) -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=list(obj.get("scheduled_new_reqs", [])),
        scheduled_cached_reqs=_to_cached_request_data(obj.get("scheduled_cached_reqs")),
        new_token_ids=obj.get("new_token_ids"),
        new_block_ids=obj.get("new_block_ids"),
        num_computed_tokens=obj.get("num_computed_tokens"),
        num_scheduled_tokens=dict(obj.get("num_scheduled_tokens", {})),
        total_num_scheduled_tokens=int(obj.get("total_num_scheduled_tokens", 0)),
        scheduled_spec_decode_tokens=dict(obj.get("scheduled_spec_decode_tokens", {})),
        scheduled_encoder_inputs=dict(obj.get("scheduled_encoder_inputs", {})),
        num_common_prefix_blocks=list(obj.get("num_common_prefix_blocks", [])),
        finished_req_ids=set(obj.get("finished_req_ids", [])),
        free_encoder_input_ids=list(obj.get("free_encoder_input_ids", [])),
        structured_output_request_ids=dict(obj.get("structured_output_request_ids", {})),
        grammar_bitmask=obj.get("grammar_bitmask"),
        kv_connector_metadata=obj.get("kv_connector_metadata"),
    )


# ---------- Combined streaming parser ----------

def parse_log_iterations(path: str) -> List[Iteration]:
    """
    Parses a log where each loop prints:
      1) a SchedulerOutput(...) line (possibly multi-line),
      2) later an EXECUTE MODEL block with context/current lengths,
      3) and finally the Execute[...] timing line that ends the iteration.

    Returns a list of Iteration(s), each pairing the SchedulerOutput that
    preceded the EXECUTE block with the Batch built from that block.
    """
    out: List[Iteration] = []

    # State for scheduler chunk
    collecting_sched = False
    sched_buf = []
    paren_depth = 0
    last_scheduler: Optional[SchedulerOutput] = None

    # State for batch block
    in_execute_block = False
    last_ctx: List[int] = []
    last_cur: List[int] = []

    def _flush_scheduler_from_buf():
        nonlocal sched_buf, last_scheduler, collecting_sched, paren_depth
        if not sched_buf:
            return
        text = "".join(sched_buf)
        try:
            lit = _literalize_constructor_expr(text)
            last_scheduler = _to_scheduler_output(lit)
        except Exception:
            # If parsing fails, we drop this scheduler; next iteration may still succeed
            last_scheduler = None
        # reset buffer/state
        sched_buf = []
        collecting_sched = False
        paren_depth = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        from tqdm import tqdm

        # First, count the number of lines for progress bar (optional, but nice UX)
        # If the file is huge, this could be slow, so we try/except and fallback to no total.
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f_count:
                total_lines = sum(1 for _ in f_count)
        except Exception:
            total_lines = None

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            iterator = tqdm(f, total=total_lines, desc="Parsing log")
            i = 0
            for raw in iterator:
                i += 1
                if i > 100000: break
                line = _strip_ansi(raw).rstrip("\n")
                # ---------- SchedulerOutput collector (multi-line) ----------
                if not collecting_sched:
                    start = line.find("SchedulerOutput(")
                    if start != -1:
                        collecting_sched = True
                        sched_buf = []
                        paren_depth = 0
                        # start collecting from 'SchedulerOutput(' (keep everything from start)
                        fragment = line[start:]
                        sched_buf.append(fragment + "\n")
                        # initialize paren depth from the fragment
                        paren_depth += fragment.count("(") - fragment.count(")")
                        # If it closed on the same line, flush immediately
                        if paren_depth == 0:
                            _flush_scheduler_from_buf()
                        # continue; also let batch parsing proceed below (same line might include arrays)
                else:
                    # Already collecting scheduler payload; continue until paren_depth returns to 0
                    sched_buf.append(line + "\n")
                    paren_depth += line.count("(") - line.count(")")
                    if paren_depth == 0:
                        _flush_scheduler_from_buf()
                    # Even while collecting scheduler, we allow batch markers on same line after flush.

                # ---------- Batch arrays ----------
                # Start of an iteration's execute block
                if "EXECUTE MODEL" in line:
                    in_execute_block = True
                    last_ctx = []
                    last_cur = []

                if in_execute_block:
                    m_ctx = CTX_RE.search(line)
                    if m_ctx:
                        last_ctx = _parse_int_list(m_ctx.group(1))

                    m_cur = CUR_RE.search(line)
                    if m_cur:
                        last_cur = _parse_int_list(m_cur.group(1))

                    # End of the iteration when timing line appears
                    m_time = EXEC_TIME_RE.search(line)
                    if m_time:
                        elapsed = float(m_time.group(1))
                        # Only emit if we have both a scheduler and the arrays
                        if last_scheduler is not None and last_ctx and last_cur:
                            batch = Batch(
                                context_lengths=last_ctx,
                                current_lengths=last_cur,
                                elasped_time=elapsed,
                            )
                            out.append(Iteration(scheduler=last_scheduler, batch=batch))
                        # Reset for next loop
                        in_execute_block = False
                        last_ctx = []
                        last_cur = []

    return out

def _ls_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Ordinary least squares with small Tikhonov for numerical stability
    # Solves min ||X w - y||_2
    ridge = 1e-9
    return np.linalg.lstsq(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ y, rcond=None)[0]

def fit(batches: List[Batch], max_iters: int = 20, tol: float = 1e-6) -> Callable[[Batch], float]:
    """
    Fits parameters a,b,c and d,e for:
        time = max( a * total_multiply + b * total_current_length + c,
                    d + e * total_length )
    Returns a callable f(batch) -> predicted_time.
    """
    if not batches:
        raise ValueError("No batches to fit.")

    # Extract features/targets
    # Be robust to "total_multiple" vs "total_multiply" naming in the prompt.
    def total_mult(b: Batch) -> float:
        return getattr(b, "total_multiple", getattr(b, "total_multiply"))

    x1 = np.array([[total_mult(b), b.total_current_length, 1.0] for b in batches], dtype=float)  # [a, b, c]
    x2 = np.array([[1.0, b.total_length] for b in batches], dtype=float)                         # [d, e]
    y  = np.array([b.elasped_time for b in batches], dtype=float)

    # Initialize by fitting both branches to y independently
    w1 = _ls_fit(x1, y)                # [a, b, c]
    w2 = _ls_fit(x2, y)                # [d, e]

    prev_obj = np.inf

    for _ in range(max_iters):
        # Compute branch predictions
        pred1 = x1 @ w1
        pred2 = x2 @ w2
        pred  = np.maximum(pred1, pred2)

        # Objective: squared error to observed times
        obj = float(np.mean((pred - y) ** 2))

        # Convergence check
        if abs(prev_obj - obj) <= tol * max(1.0, prev_obj):
            break
        prev_obj = obj

        # Assign active branch per sample
        active1 = pred1 >= pred2
        active2 = ~active1

        # Refit each branch **only on samples where it is active**.
        # If one branch has too few points, fall back to global fit.
        if np.sum(active1) >= 3:
            w1 = _ls_fit(x1[active1], y[active1])
        else:
            w1 = _ls_fit(x1, y)

        if np.sum(active2) >= 3:
            w2 = _ls_fit(x2[active2], y[active2])
        else:
            w2 = _ls_fit(x2, y)

    # Build the predictor
    a, b, c = w1.tolist()
    d, e    = w2.tolist()

    def predictor(batch: Batch) -> float:
        tmul = getattr(batch, "total_multiple", getattr(batch, "total_multiply"))
        v1 = a * tmul + b * batch.total_current_length + c
        v2 = d + e * batch.total_length
        return max(v1, v2)

    return predictor, (a, b, c, d, e)

iterations = parse_log_iterations('vllm_new.log')

batches = [iteration.batch for iteration in iterations]

predictor, (a, b, c, d, e) = fit(batches)

print('fitted_model: ')
print(f'time = max({a:.6f} * total_multiply + {b:.6f} * total_current_length + {c:.6f}, {d:.6f} + {e:.6f} * total_length)')

predicted_times = [predictor(batch) for batch in batches]

import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(3, figsize = (4, 9), tight_layout = True)

current_lengths = [batch.total_current_length for batch in batches]
context_lengths = [batch.average_context_length for batch in batches]
elapsed_times = [batch.elasped_time for batch in batches]

ax1.scatter(current_lengths, elapsed_times)
ax1.set_xlabel('Current Length')
ax1.set_ylabel('Elapsed Time')

ax2.scatter(context_lengths, elapsed_times)
ax2.set_xlabel('Context Length')
ax2.set_ylabel('Elapsed Time')

ax3.scatter(elapsed_times, predicted_times)
ax3.set_xlabel('Elapsed Time')
ax3.set_ylabel('Predicted Time')

fig.savefig('figs/vllm_log.png')

num_scheduled_new_reqs = [len(iteration.scheduler.scheduled_new_reqs) for iteration in iterations]

print('total number of new requests: ', sum(num_scheduled_new_reqs))
print('average number of new requests: ', sum(num_scheduled_new_reqs) / len(num_scheduled_new_reqs))

# exit()
for i in range(10):
    print(f'Sample Iteration #{i}:')
    print(iterations[i].scheduler.scheduled_new_reqs)

def iterations_to_requests(iterations: List[Iteration]) -> List[Request]:
    requests = {}
    t = 0
    for idx, iteration in enumerate(iterations):
        scheduler = iteration.scheduler
        t_after = t + iteration.batch.elasped_time
        
        for req_id, num_computed_tokens in zip(
            scheduler.scheduled_cached_reqs.req_ids,
            scheduler.scheduled_cached_reqs.num_computed_tokens
        ):
            if req_id not in requests:
                requests[req_id] = Request(request_id=req_id)
                requests[req_id].timestamps.append(t)
            requests[req_id].timestamps.append(t_after)
            requests[req_id].iterations.append((idx, 
                                                num_computed_tokens,
                                                iteration.scheduler.num_scheduled_tokens[req_id],
                                                iteration.batch.elasped_time))
            
            
    return requests


requests = iterations_to_requests(iterations)

n_to_display = 100
for req_id, request in requests.items():
    print(f'Request {req_id}:')
    print(f'  Timestamps: {request.timestamps}')
    print(f'  Iterations: {request.iterations}')
    n_to_display -= 1 
    if n_to_display < 0: break 
    
