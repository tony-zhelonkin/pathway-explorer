"""
HTML generation module for Pathway Explorer.

Handles:
- Data preparation for JSON export
- CSS styles generation
- JavaScript logic generation
- Complete HTML template assembly
"""

import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .config import COLORS, DB_COLORS, NES_MAX, PLOTLY_JS_URL


def prepare_pathway_data(
    df: pd.DataFrame,
    embedding: np.ndarray,
    neighbors: Dict[str, List]
) -> List[Dict[str, Any]]:
    """
    Prepare pathway data for JSON export to the dashboard.

    Args:
        df: DataFrame with pathway data (must have 'genes' as sets)
        embedding: 2D coordinates from dimensionality reduction
        neighbors: Dict of pathway_id -> neighbor list

    Returns:
        List of pathway dictionaries for JSON serialization
    """
    pathways = []

    # Check which columns exist
    has_leading_edge = 'leading_edge_size' in df.columns

    for i, (_, row) in enumerate(df.iterrows()):
        # Calculate leading_edge_size from genes if not present
        if has_leading_edge:
            leading_edge_size = int(row['leading_edge_size']) if pd.notna(row['leading_edge_size']) else 0
        else:
            leading_edge_size = len(row['genes']) if 'genes' in row.index else 0

        p = {
            'id': row['pathway_id'],
            'name': row['display_name'],
            'full_name': row['pathway_name'],
            'description': row.get('pathway_name', ''),
            'database': row['database'],
            'entity_type': row.get('entity_type', 'Pathway'),
            'nes': round(float(row['nes']), 3) if pd.notna(row['nes']) else 0,
            'signed_sig': round(float(row['signed_sig']), 2) if pd.notna(row.get('signed_sig')) else 0,
            'padj': float(row['padj']) if pd.notna(row['padj']) else 1,
            'pvalue': float(row['pvalue']) if pd.notna(row['pvalue']) else 1,
            'set_size': int(row['set_size']) if pd.notna(row['set_size']) else 0,
            'leading_edge_size': leading_edge_size,
            'direction': row.get('direction', 'Up' if row['nes'] > 0 else 'Down'),
            'x': round(float(embedding[i, 0]), 4),
            'y': round(float(embedding[i, 1]), 4),
            'genes': '/'.join(sorted(list(row['genes'])[:20])),  # Top 20 genes
            'gene_count': len(row['genes']),
            'neighbors': neighbors.get(row['pathway_id'], [])
        }
        pathways.append(p)

    return pathways


def generate_html(pathways: List[Dict], metadata: Dict) -> str:
    """
    Generate the complete HTML dashboard.

    Args:
        pathways: List of pathway dictionaries
        metadata: Dashboard metadata (databases, embedding method, etc.)

    Returns:
        Complete HTML string
    """
    pathways_json = json.dumps(pathways, indent=None)
    metadata_json = json.dumps(metadata, indent=None)

    css = _generate_css()
    js = _generate_javascript(pathways_json, metadata_json)

    contrast = metadata.get('contrast', 'All')
    title = f"Pathway Explorer - {contrast}" if contrast != 'All' else "Pathway Explorer"

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="{PLOTLY_JS_URL}"></script>
    <style>
{css}
    </style>
</head>
<body>

{_generate_sidebar_html(metadata)}

{_generate_main_content_html()}

<script>
{js}
</script>
</body>
</html>'''


def _generate_css() -> str:
    """Generate CSS styles for the dashboard."""
    return f'''        :root {{
            --primary: #2c3e50;
            --accent: #3498db;
            --bg: #f8f9fa;
            --sidebar-width: 320px;
        }}

        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: var(--bg);
            display: flex;
            height: 100vh;
            overflow: hidden;
        }}

        .sidebar {{
            width: var(--sidebar-width);
            background: white;
            border-right: 1px solid #ddd;
            padding: 16px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 16px;
            box-shadow: 2px 0 8px rgba(0,0,0,0.05);
        }}

        .sidebar h1 {{
            font-size: 1.1em;
            margin: 0 0 4px 0;
            color: var(--primary);
        }}

        .sidebar .subtitle {{
            font-size: 0.8em;
            color: #666;
            margin-bottom: 8px;
        }}

        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}

        .control-group label {{
            font-weight: 600;
            font-size: 0.85em;
            color: var(--primary);
        }}

        .checkbox-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            max-height: 120px;
            overflow-y: auto;
            border: 1px solid #eee;
            padding: 6px;
            border-radius: 4px;
            background: #fafafa;
        }}

        .checkbox-item {{
            display: flex;
            align-items: center;
            gap: 4px;
            font-size: 0.8em;
            padding: 2px 6px;
            border-radius: 3px;
            cursor: pointer;
        }}
        .checkbox-item:hover {{ background: #e8f4fc; }}

        input[type="text"], input[type="number"], select {{
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            width: 100%;
            font-size: 0.9em;
        }}

        input[type="range"] {{ width: 100%; }}

        .range-row {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .range-value {{
            min-width: 40px;
            text-align: right;
            font-size: 0.85em;
            color: #666;
        }}

        .btn {{
            padding: 8px 12px;
            background: var(--accent);
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85em;
            transition: background 0.2s;
        }}
        .btn:hover {{ background: #2980b9; }}
        .btn-sm {{ padding: 4px 8px; font-size: 0.75em; }}

        .main {{
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 16px;
            gap: 8px;
            min-width: 0;
        }}

        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 8px;
            border-bottom: 1px solid #eee;
        }}

        .stats {{ font-size: 0.85em; color: #666; }}

        .chart-container {{
            flex: 1;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            min-height: 0;
            overflow: hidden;
            position: relative;
        }}

        #main-chart {{
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
        }}

        .detail-panel {{
            background: white;
            border-radius: 8px;
            padding: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            min-height: 180px;
            max-height: 200px;
            overflow-y: auto;
            flex-shrink: 0;
        }}

        .detail-panel h3 {{
            margin: 0 0 8px 0;
            font-size: 0.95em;
            color: var(--primary);
        }}

        .detail-row {{
            display: flex;
            gap: 16px;
            font-size: 0.8em;
            margin-bottom: 4px;
        }}

        .detail-label {{ color: #666; min-width: 80px; }}
        .detail-value {{ color: #333; font-weight: 500; }}

        .gene-list {{
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 0.75em;
            color: #555;
            line-height: 1.4;
            max-height: 60px;
            overflow-y: auto;
        }}

        .neighbors-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            margin-top: 8px;
        }}

        .neighbor-chip {{
            background: #e8f4fc;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.75em;
            color: var(--accent);
            cursor: pointer;
        }}
        .neighbor-chip:hover {{ background: #d0e8f7; }}

        hr {{ border: 0; border-top: 1px solid #eee; margin: 8px 0; }}

        .legend {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.75em;
            padding: 8px;
            background: #f5f5f5;
            border-radius: 4px;
        }}

        .legend-gradient {{
            width: 100px;
            height: 12px;
            background: linear-gradient(to right, {COLORS['negative']}, {COLORS['neutral']}, {COLORS['positive']});
            border-radius: 2px;
        }}

        .hidden {{ display: none !important; }}

        @keyframes pathway-pulse {{
            0%, 100% {{ transform: scale(1); opacity: 1; }}
            25% {{ transform: scale(1.3); opacity: 0.8; }}
            50% {{ transform: scale(1.1); opacity: 1; }}
            75% {{ transform: scale(1.2); opacity: 0.9; }}
        }}

        .pathway-highlight {{
            animation: pathway-pulse 0.8s ease-in-out;
        }}'''


def _generate_sidebar_html(metadata: Dict = None) -> str:
    """Generate sidebar HTML with controls."""
    contrast = metadata.get('contrast', 'All') if metadata else 'All'
    subtitle = f"Contrast: {contrast}" if contrast != 'All' else "GSEA Results - Gene Overlap Embedding"

    return f'''<div class="sidebar">
    <div>
        <h1>Pathway Explorer</h1>
        <div class="subtitle">{subtitle}</div>
    </div>

    <div class="control-group">
        <label>Search Pathways</label>
        <input type="text" id="search-box" placeholder="Type to filter..." oninput="updateChart()">
    </div>

    <div class="control-group">
        <label>Filter: Databases</label>
        <div class="checkbox-list" id="db-list"></div>
        <button class="btn btn-sm" onclick="toggleAll('db-list')">Toggle All</button>
    </div>

    <div class="control-group">
        <label>Significance Threshold (FDR)</label>
        <div class="range-row">
            <input type="range" id="fdr-slider" min="-3" max="0" step="0.1" value="-1.3" oninput="updateFdrDisplay(); updateChart()">
            <span class="range-value" id="fdr-value">0.05</span>
        </div>
    </div>

    <div class="control-group">
        <label>Min |NES|</label>
        <div class="range-row">
            <input type="range" id="nes-slider" min="0" max="3" step="0.1" value="0" oninput="updateNesDisplay(); updateChart()">
            <span class="range-value" id="nes-value">0.0</span>
        </div>
    </div>

    <div class="control-group">
        <label>Direction Filter</label>
        <select id="direction-filter" onchange="updateChart()">
            <option value="all">All</option>
            <option value="up">Upregulated (NES > 0)</option>
            <option value="down">Downregulated (NES < 0)</option>
        </select>
    </div>

    <div class="control-group">
        <label>Entity Type</label>
        <select id="entity-filter" onchange="updateChart()">
            <option value="all">All Entity Types</option>
            <option value="pathway">Pathways only (GSEA)</option>
            <option value="tf">TFs only (CollecTRI)</option>
            <option value="progeny">PROGENy only (14 signaling pathways)</option>
            <option value="te">TEs only (Transposable Elements)</option>
        </select>
    </div>

    <div class="control-group" id="te-level-group" style="display:none;">
        <label>TE Aggregation Level</label>
        <select id="te-level-filter" onchange="updateChart()">
            <option value="all">All TE Levels</option>
            <option value="family" selected>Family Level</option>
            <option value="class">Class Level</option>
        </select>
    </div>

    <hr>

    <div class="control-group">
        <label>Display Options</label>
        <div class="checkbox-item">
            <input type="checkbox" id="show-edges" onchange="updateChart()">
            <span>Show gene overlap edges</span>
        </div>
        <div class="checkbox-item">
            <input type="checkbox" id="color-by-db" onchange="updateChart()">
            <span>Color by database</span>
        </div>
        <div class="checkbox-item">
            <input type="checkbox" id="use-unified-score" checked onchange="updateChart(); updateLegend();">
            <span>Use unified score (TF+Pathway comparable)</span>
        </div>
    </div>

    <div class="legend" id="color-legend">
        <span id="legend-label">Score:</span>
        <span style="color:{COLORS['negative']}">Down</span>
        <div class="legend-gradient"></div>
        <span style="color:{COLORS['positive']}">Up</span>
    </div>

    <div class="legend" style="margin-top: 4px;">
        <span>Shape:</span>
        <span>&#9679; Pathway</span>
        <span>&#9670; TF</span>
        <span>&#9632; PROGENy</span>
        <span>&#9650; TE</span>
    </div>

    <div style="margin-top: auto; display: flex; flex-direction: column; gap: 8px;">
        <button class="btn" onclick="exportFilteredData()" style="width: 100%;">Export Filtered CSV</button>
        <button class="btn" onclick="resetFilters()" style="width: 100%;">Reset All Filters</button>
    </div>
</div>'''


def _generate_main_content_html() -> str:
    """Generate main content area HTML."""
    return '''<div class="main">
    <div class="header">
        <span class="stats" id="status-bar">Loading...</span>
    </div>

    <div class="chart-container">
        <div id="main-chart"></div>
    </div>

    <div class="detail-panel" id="detail-panel">
        <h3>Pathway Details</h3>
        <p style="color: #999; font-size: 0.85em;">Click a pathway to see details and neighbors</p>
    </div>
</div>'''


def _generate_javascript(pathways_json: str, metadata_json: str) -> str:
    """Generate JavaScript logic for the dashboard."""
    return f'''// ============================================================================
// DATA
// ============================================================================
const RAW_DATA = {pathways_json};
const METADATA = {metadata_json};

const DB_COLORS = {json.dumps(DB_COLORS)};
const NES_COLORS = {json.dumps(COLORS)};
const NES_MAX = {NES_MAX};

let filteredData = [];
let selectedPathway = null;

// ============================================================================
// INITIALIZATION
// ============================================================================
function init() {{
    populateDbList();
    updateChart();
}}

function populateDbList() {{
    const container = document.getElementById('db-list');
    container.innerHTML = '';

    METADATA.databases.forEach(db => {{
        const div = document.createElement('div');
        div.className = 'checkbox-item';
        const color = DB_COLORS[db] || '#999';
        div.innerHTML = `
            <input type="checkbox" value="${{db}}" checked onchange="updateChart()">
            <span style="color:${{color}}">&#9679;</span>
            <span>${{db}}</span>
        `;
        container.appendChild(div);
    }});
}}

function toggleAll(id) {{
    const inputs = document.querySelectorAll(`#${{id}} input[type="checkbox"]`);
    const allChecked = Array.from(inputs).every(i => i.checked);
    inputs.forEach(i => i.checked = !allChecked);
    updateChart();
}}

function resetFilters() {{
    document.getElementById('search-box').value = '';
    document.getElementById('fdr-slider').value = -1.3;
    document.getElementById('nes-slider').value = 0;
    document.getElementById('direction-filter').value = 'all';
    document.getElementById('entity-filter').value = 'all';
    const teLevelEl = document.getElementById('te-level-filter');
    if (teLevelEl) teLevelEl.value = 'family';
    document.getElementById('show-edges').checked = false;
    document.getElementById('color-by-db').checked = false;
    document.getElementById('use-unified-score').checked = true;
    document.querySelectorAll('#db-list input').forEach(i => i.checked = true);
    updateFdrDisplay();
    updateNesDisplay();
    updateLegend();
    updateChart();
}}

function updateFdrDisplay() {{
    const val = Math.pow(10, parseFloat(document.getElementById('fdr-slider').value));
    document.getElementById('fdr-value').textContent = val.toFixed(val < 0.01 ? 3 : 2);
}}

function updateNesDisplay() {{
    const val = parseFloat(document.getElementById('nes-slider').value);
    document.getElementById('nes-value').textContent = val.toFixed(1);
}}

// ============================================================================
// CHART RENDERING
// ============================================================================
function updateChart() {{
    const settings = getSettings();

    // Show/hide TE level filter based on entity filter
    const teGroup = document.getElementById('te-level-group');
    if (teGroup) {{
        teGroup.style.display = (settings.entityFilter === 'te' || settings.entityFilter === 'all') ? 'flex' : 'none';
    }}

    filteredData = RAW_DATA.filter(d => {{
        if (!settings.databases.includes(d.database)) return false;
        if (d.padj > settings.fdrThreshold) return false;
        if (Math.abs(d.nes) < settings.minNes) return false;
        if (settings.direction === 'up' && d.nes <= 0) return false;
        if (settings.direction === 'down' && d.nes >= 0) return false;
        if (settings.entityFilter === 'tf' && d.entity_type !== 'TF') return false;
        if (settings.entityFilter === 'pathway' && d.entity_type !== 'Pathway') return false;
        if (settings.entityFilter === 'progeny' && d.entity_type !== 'PROGENy') return false;
        if (settings.entityFilter === 'te' && d.entity_type !== 'TE') return false;
        // TE level filter (Family vs Class)
        if (d.entity_type === 'TE' && settings.teLevel !== 'all') {{
            const dbLevel = d.database.replace('TE_', '').toLowerCase();
            if (dbLevel !== settings.teLevel) return false;
        }}
        if (settings.search && !d.name.toLowerCase().includes(settings.search) &&
            !d.full_name.toLowerCase().includes(settings.search)) return false;
        return true;
    }});

    const nTFs = filteredData.filter(d => d.entity_type === 'TF').length;
    const nPROGENy = filteredData.filter(d => d.entity_type === 'PROGENy').length;
    const nPathways = filteredData.filter(d => d.entity_type === 'Pathway').length;
    const nTEs = filteredData.filter(d => d.entity_type === 'TE').length;
    document.getElementById('status-bar').textContent =
        `Showing ${{filteredData.length}} of ${{RAW_DATA.length}} (${{nPathways}} pathways, ${{nTFs}} TFs, ${{nPROGENy}} PROGENy, ${{nTEs}} TEs)`;

    renderScatter(settings);
}}

function getSettings() {{
    const teLevelEl = document.getElementById('te-level-filter');
    return {{
        databases: Array.from(document.querySelectorAll('#db-list input:checked')).map(i => i.value),
        fdrThreshold: Math.pow(10, parseFloat(document.getElementById('fdr-slider').value)),
        minNes: parseFloat(document.getElementById('nes-slider').value),
        direction: document.getElementById('direction-filter').value,
        entityFilter: document.getElementById('entity-filter').value,
        teLevel: teLevelEl ? teLevelEl.value : 'all',
        search: document.getElementById('search-box').value.toLowerCase(),
        showEdges: document.getElementById('show-edges').checked,
        colorByDb: document.getElementById('color-by-db').checked,
        useUnifiedScore: document.getElementById('use-unified-score').checked
    }};
}}

function updateLegend() {{
    const useUnified = document.getElementById('use-unified-score').checked;
    const label = document.getElementById('legend-label');
    label.textContent = useUnified ? 'Unified:' : 'NES:';
}}

function exportFilteredData() {{
    if (filteredData.length === 0) {{
        alert('No data to export. Adjust filters to show some results.');
        return;
    }}

    const headers = ['id', 'name', 'entity_type', 'database', 'nes', 'signed_sig', 'padj', 'direction', 'set_size', 'leading_edge_size', 'gene_count'];
    const rows = [headers.join(',')];

    filteredData.forEach(d => {{
        const row = [
            `"${{d.id}}"`,
            `"${{d.name}}"`,
            d.entity_type,
            d.database,
            d.nes,
            d.signed_sig,
            d.padj,
            d.direction,
            d.set_size,
            d.leading_edge_size,
            d.gene_count
        ];
        rows.push(row.join(','));
    }});

    const csvContent = rows.join('\\n');
    const blob = new Blob([csvContent], {{ type: 'text/csv;charset=utf-8;' }});
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `pathway_explorer_filtered_${{filteredData.length}}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}}

function scoreToColor(score, useUnified) {{
    if (score === null || isNaN(score)) return '#999';
    const maxVal = useUnified ? 30 : NES_MAX;
    const t = Math.max(-1, Math.min(1, score / maxVal));
    if (t <= 0) {{
        return interpolateColor(NES_COLORS.negative, NES_COLORS.neutral, 1 + t);
    }} else {{
        return interpolateColor(NES_COLORS.neutral, NES_COLORS.positive, t);
    }}
}}

function interpolateColor(c1, c2, t) {{
    const r1 = parseInt(c1.slice(1,3), 16);
    const g1 = parseInt(c1.slice(3,5), 16);
    const b1 = parseInt(c1.slice(5,7), 16);
    const r2 = parseInt(c2.slice(1,3), 16);
    const g2 = parseInt(c2.slice(3,5), 16);
    const b2 = parseInt(c2.slice(5,7), 16);
    const r = Math.round(r1 + (r2 - r1) * t);
    const g = Math.round(g1 + (g2 - g1) * t);
    const b = Math.round(b1 + (b2 - b1) * t);
    return `#${{r.toString(16).padStart(2,'0')}}${{g.toString(16).padStart(2,'0')}}${{b.toString(16).padStart(2,'0')}}`;
}}

function renderScatter(settings) {{
    const traces = [];

    const x = filteredData.map(d => d.x);
    const y = filteredData.map(d => d.y);

    let colors;
    if (settings.colorByDb) {{
        colors = filteredData.map(d => DB_COLORS[d.database] || '#999');
    }} else {{
        colors = filteredData.map(d => {{
            const score = settings.useUnifiedScore ? d.signed_sig : d.nes;
            return scoreToColor(score, settings.useUnifiedScore);
        }});
    }}

    const sizes = filteredData.map(d => {{
        const negLogP = -Math.log10(Math.max(d.padj, 1e-50));
        return Math.min(Math.max(negLogP * 2, 6), 30);
    }});

    const hoverTexts = filteredData.map(d => {{
        const dir = d.nes > 0 ? 'Up' : 'Down';
        const typeLabels = {{'TF': 'TF', 'PROGENy': 'PROGENy', 'TE': 'TE', 'Pathway': 'Pathway'}};
        const typeLabel = typeLabels[d.entity_type] || 'Pathway';
        const geneLabel = d.entity_type === 'TE' ? 'Subfamilies' : 'Genes';
        return `<b>${{d.name}}</b> (${{typeLabel}})<br>` +
               `Database: ${{d.database}}<br>` +
               `NES/Activity: ${{d.nes.toFixed(2)}} (${{dir}})<br>` +
               `Unified Score: ${{d.signed_sig.toFixed(1)}}<br>` +
               `FDR: ${{d.padj.toExponential(2)}}<br>` +
               `${{geneLabel}}: ${{d.gene_count}} (${{d.leading_edge_size}} in core)`;
    }});

    const symbols = filteredData.map(d => {{
        if (d.entity_type === 'TF') return 'diamond';
        if (d.entity_type === 'PROGENy') return 'square';
        if (d.entity_type === 'TE') return 'triangle-up';
        return 'circle';
    }});

    traces.push({{
        x: x,
        y: y,
        mode: 'markers',
        type: 'scattergl',
        marker: {{
            color: colors,
            size: sizes,
            symbol: symbols,
            opacity: 0.7,
            line: {{ color: 'white', width: 1 }}
        }},
        text: hoverTexts,
        hoverinfo: 'text',
        customdata: filteredData.map(d => d.id)
    }});

    if (settings.showEdges && filteredData.length < 500) {{
        const edgeX = [];
        const edgeY = [];
        const filteredIds = new Set(filteredData.map(d => d.id));

        filteredData.forEach(d => {{
            if (d.neighbors) {{
                d.neighbors.forEach(([neighborId, sim]) => {{
                    if (filteredIds.has(neighborId)) {{
                        const neighbor = filteredData.find(n => n.id === neighborId);
                        if (neighbor) {{
                            edgeX.push(d.x, neighbor.x, null);
                            edgeY.push(d.y, neighbor.y, null);
                        }}
                    }}
                }});
            }}
        }});

        if (edgeX.length > 0) {{
            traces.unshift({{
                x: edgeX,
                y: edgeY,
                mode: 'lines',
                type: 'scatter',
                line: {{ color: 'rgba(150, 150, 150, 0.2)', width: 1 }},
                hoverinfo: 'skip'
            }});
        }}
    }}

    const layout = {{
        margin: {{ t: 20, b: 40, l: 40, r: 20 }},
        showlegend: false,
        xaxis: {{
            title: 'UMAP 1 (Gene Overlap)',
            showgrid: true,
            gridcolor: '#f0f0f0',
            zeroline: false
        }},
        yaxis: {{
            title: 'UMAP 2 (Gene Overlap)',
            showgrid: true,
            gridcolor: '#f0f0f0',
            zeroline: false
        }},
        hovermode: 'closest',
        hoverlabel: {{
            bgcolor: 'white',
            bordercolor: '#ccc',
            font: {{ size: 11 }}
        }},
        dragmode: 'pan'
    }};

    const config = {{
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        scrollZoom: true
    }};

    Plotly.react('main-chart', traces, layout, config);

    document.getElementById('main-chart').on('plotly_click', function(data) {{
        if (data.points && data.points[0]) {{
            const pathwayId = data.points[0].customdata;
            showPathwayDetails(pathwayId);
            highlightPathwayMarker(pathwayId);
        }}
    }});
}}

// ============================================================================
// CENTERING AND HIGHLIGHTING
// ============================================================================
function centerOnPathway(pathwayId, animate = true) {{
    const pathway = filteredData.find(d => d.id === pathwayId);
    if (!pathway) return;

    const padding = 0.15;
    const xRange = [pathway.x - padding, pathway.x + padding];
    const yRange = [pathway.y - padding, pathway.y + padding];

    Plotly.relayout('main-chart', {{
        'xaxis.range': xRange,
        'yaxis.range': yRange
    }}).then(() => {{
        setTimeout(() => {{
            highlightPathwayMarker(pathwayId);
        }}, 50);
    }});
}}

function highlightPathwayMarker(pathwayId) {{
    const pathwayIndex = filteredData.findIndex(d => d.id === pathwayId);
    if (pathwayIndex === -1) return;

    const chartDiv = document.getElementById('main-chart');
    const chartData = chartDiv.data || [];
    let scatterTraceIndex = 0;

    for (let i = 0; i < chartData.length; i++) {{
        if (chartData[i].customdata && chartData[i].customdata.length > 0) {{
            scatterTraceIndex = i;
            break;
        }}
    }}

    const currentSizes = filteredData.map(d => {{
        const negLogP = -Math.log10(Math.max(d.padj, 1e-50));
        return Math.min(Math.max(negLogP * 2, 6), 30);
    }});

    const highlightSizes = currentSizes.slice();
    const originalSize = highlightSizes[pathwayIndex];

    const pulseFrames = [
        {{ size: originalSize * 2.5 }},
        {{ size: originalSize * 1.8 }},
        {{ size: originalSize }}
    ];

    let frameIndex = 0;
    const animateFrame = () => {{
        if (frameIndex >= pulseFrames.length) return;

        highlightSizes[pathwayIndex] = pulseFrames[frameIndex].size;

        Plotly.restyle('main-chart', {{
            'marker.size': [highlightSizes]
        }}, [scatterTraceIndex]);

        frameIndex++;
        if (frameIndex < pulseFrames.length) {{
            setTimeout(animateFrame, 150);
        }}
    }};

    animateFrame();
}}

function navigateToPathway(pathwayId) {{
    centerOnPathway(pathwayId, true);
    showPathwayDetails(pathwayId);
}}

// ============================================================================
// DETAIL PANEL
// ============================================================================
function showPathwayDetails(pathwayId) {{
    const pathway = RAW_DATA.find(d => d.id === pathwayId);
    if (!pathway) return;

    selectedPathway = pathway;

    const panel = document.getElementById('detail-panel');
    const dir = pathway.nes > 0 ? 'Upregulated' : 'Downregulated';
    const dirColor = pathway.nes > 0 ? '{COLORS["positive"]}' : '{COLORS["negative"]}';

    let neighborsHtml = '';
    if (pathway.neighbors && pathway.neighbors.length > 0) {{
        neighborsHtml = `
            <div style="margin-top: 8px;">
                <strong style="font-size: 0.8em;">Similar pathways (by gene overlap) - click to navigate:</strong>
                <div class="neighbors-list">
                    ${{pathway.neighbors.map(([id, sim]) => {{
                        const n = RAW_DATA.find(d => d.id === id);
                        const name = n ? n.name.slice(0, 30) : id.slice(0, 30);
                        return `<span class="neighbor-chip" onclick="navigateToPathway('${{id}}')" title="Click to center view on this pathway">${{name}} (${{(sim*100).toFixed(0)}}%)</span>`;
                    }}).join('')}}
                </div>
            </div>
        `;
    }}

    panel.innerHTML = `
        <h3 style="color: ${{dirColor}}">${{pathway.name}}</h3>
        <div class="detail-row">
            <span class="detail-label">Database:</span>
            <span class="detail-value">${{pathway.database}}</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">NES:</span>
            <span class="detail-value" style="color: ${{dirColor}}">${{pathway.nes.toFixed(3)}} (${{dir}})</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">FDR:</span>
            <span class="detail-value">${{pathway.padj.toExponential(2)}}</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Set size:</span>
            <span class="detail-value">${{pathway.set_size}} genes (${{pathway.leading_edge_size}} in leading edge)</span>
        </div>
        <div style="margin-top: 8px;">
            <strong style="font-size: 0.8em;">Top leading edge genes:</strong>
            <div class="gene-list">${{pathway.genes.replace(/\\//g, ', ')}}</div>
        </div>
        ${{neighborsHtml}}
    `;
}}

// Initialize on load
init();
updateFdrDisplay();
updateNesDisplay();
updateLegend();'''
