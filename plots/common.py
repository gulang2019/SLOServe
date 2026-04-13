from pathlib import Path


LABEL_MAP = {
    'atfc / round_robin': 'SLO Packer (RR)',
    'atfc / round_robin_session': 'SLO Packer (Session-Aware)',
    'vllm / round_robin_session': 'vLLM (Session-Aware)',
    'sarathi / round_robin_session': 'Sarathi+ (Session-Aware)',
    'atfc / slosserve_planner_ablation-no_local': 'SLO Packer (no local)',
    'atfc / slosserve_planner_ablation-no_global': 'SLO Packer (no global)',
    'atfc / slosserve_planner': 'SLO Packer',
    'atfc / slosserve_planner_oracle_mem': 'SLO Packer (Oracle)',
    'qlm / round_robin': 'vLLM+',
    'vllm / round_robin': 'vLLM',
    'sarathi+ / round_robin': 'Sarathi+',
    'sarathi / round_robin': 'Sarathi',
    'qlm / llumnix_load': 'Llumnix',
    'atfc / slosserve_disagg_planner': 'SLO Packer (Disagg)',
    'qlm / round_robin-disagg': 'vLLM+ (Disagg)'
}

STYLE_KEY_ALIASES = {
    'Baseline': 'qlm / round_robin',
    'Ours': 'atfc / slosserve_planner',
    'SLO Packer': 'atfc / slosserve_planner',
    'SLO-Packer': 'atfc / slosserve_planner',
    'SLO Packer (RR)': 'atfc / round_robin',
    'SLO-Packer (RR)': 'atfc / round_robin',
    'SLO Packer (Oracle)': 'atfc / slosserve_planner_oracle_mem',
    'SLO-Packer (Oracle)': 'atfc / slosserve_planner_oracle_mem',
    'QLM': 'qlm / round_robin',
    'vLLM+': 'vllm / round_robin',
    'vLLM (Session-Aware)': 'vllm / round_robin_session',
    'Sarathi+': 'sarathi+ / round_robin',
    'Sarathi+ (Session-Aware)': 'sarathi / round_robin_session',
    'Sarathi': 'sarathi / round_robin',
    'llumnix': 'qlm / llumnix_load',
}

COLOR_MAP = {
    'atfc / round_robin': '#6BAE3F',
    'atfc / slosserve_planner_ablation-no_local': '#8BCB69',
    'atfc / slosserve_planner_ablation-no_global': '#4F9A41',
    'atfc / slosserve_planner': '#2E7D32',
    'atfc / slosserve_planner_oracle_mem': '#145A32',
    'qlm / round_robin': '#E76F51',
    'vllm / round_robin': '#4C78A8',
    'vllm / round_robin_session': '#1F5AA6',
    'sarathi+ / round_robin': '#7A5195',
    'sarathi / round_robin': '#B279A2',
    'sarathi / round_robin_session': '#C44E52',
    'qlm / llumnix_load': '#F58518',
}

MARKER_MAP = {
    'atfc / round_robin': 'D',
    'atfc / slosserve_planner_ablation-no_local': 'P',
    'atfc / slosserve_planner_ablation-no_global': 'X',
    'atfc / slosserve_planner': 'o',
    'atfc / slosserve_planner_oracle_mem': '*',
    'qlm / round_robin': 's',
    'vllm / round_robin': '^',
    'vllm / round_robin_session': 'p',
    'sarathi+ / round_robin': '<',
    'sarathi / round_robin': 'v',
    'sarathi / round_robin_session': 'X',
    'qlm / llumnix_load': '>',
}

_DEFAULT_SLO_PACKER_COLOR = '#2E7D32'
_DEFAULT_BASELINE_COLOR = '#E76F51'
_DEFAULT_SLO_PACKER_MARKER = 'o'
_DEFAULT_BASELINE_MARKER = 's'
PAPER_FIGS_DIR = Path("Paper/figs")


def get_method_label(method: str) -> str:
    return LABEL_MAP.get(method, method)


def get_method_color(method: str) -> str:
    style_key = STYLE_KEY_ALIASES.get(method, method)
    if style_key in COLOR_MAP:
        return COLOR_MAP[style_key]
    if style_key.startswith('atfc /'):
        return _DEFAULT_SLO_PACKER_COLOR
    return _DEFAULT_BASELINE_COLOR


def get_method_marker(method: str) -> str:
    style_key = STYLE_KEY_ALIASES.get(method, method)
    if style_key in MARKER_MAP:
        return MARKER_MAP[style_key]
    if style_key.startswith('atfc /'):
        return _DEFAULT_SLO_PACKER_MARKER
    return _DEFAULT_BASELINE_MARKER


def get_method_style(method: str) -> dict[str, object]:
    return {
        'color': get_method_color(method),
        'marker': get_method_marker(method),
        'linewidth': 2.0,
        'markersize': 6,
    }


def get_paper_figure_dir(script_name: str, function_name: str | None = None) -> Path:
    outdir = PAPER_FIGS_DIR / script_name
    if function_name is not None:
        outdir = outdir / function_name
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir
