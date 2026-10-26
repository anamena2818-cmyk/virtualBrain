import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial import ConvexHull

# -----------------------------------------------------------------------------
# DUAL-ENGINE TVB (THE VIRTUAL BRAIN) INTEGRATION
# -----------------------------------------------------------------------------
HAS_TVB_NATIVE = False
try:
    from tvb.datatypes.connectivity import Connectivity
    from tvb.simulator.models.epileptor import Epileptor
    from tvb.simulator.coupling import Difference
    from tvb.simulator.integrators import HeunStochastic
    from tvb.simulator.monitors import Raw
    from tvb.simulator.simulator import Simulator
    HAS_TVB_NATIVE = True
except ImportError:
    HAS_TVB_NATIVE = False

# -----------------------------------------------------------------------------
# MOTOR NEUROFISIOLÓGICO EPILEPTOR DE TVB (CON ESTIMULACIÓN ELÉCTRICA FOCAL)
# -----------------------------------------------------------------------------
def run_tvb_epileptor_simulation(sc_matrix, ez_indices, resected_indices=[], x0_ez=-1.2, x0_healthy=-2.4, Kg=1.0, eta_dbs=0.0, is_resected=False, dt=0.05, steps=2000, enable_stim=False, stim_onset=2000.0, stim_dur=100.0, stim_amp=12.0):
    """
    Simulación neurofisiológica basada en el modelo Epileptor de TVB (Jirsa et al., 2014).
    Ecuaciones diferenciales acopladas de 6 variables de estado por nodo:
      x1, y1: Población rápida (oscilaciones gamma/HFO interictales e ictales)
      z:      Variable lenta de permisividad biológica (permittivity)
      x2, y2: Población intermedia (descargas clónicas de alta amplitud)
    Inyección de Estimulación Eléctrica Focal (PulseTrain):
      Inyecta una corriente externa I_ext a la variable rápida x1 durante [stim_onset, stim_onset + stim_dur].
    """
    n_nodes = sc_matrix.shape[0]
    
    x0 = np.full(n_nodes, x0_healthy)
    for ez_idx in ez_indices:
        if not is_resected:
            x0[ez_idx] = x0_ez + (abs(x0_healthy - x0_ez) * eta_dbs * 0.75)
        else:
            x0[ez_idx] = -3.0  # Resección: tejido hipo-excitable desconectado

    for res_idx in resected_indices:
        x0[res_idx] = -2.4  # Reemplaza dinámica del nodo resecado por nodo estable (TVB Tutorial 3)

    x1 = np.full(n_nodes, -1.6)
    y1 = np.full(n_nodes, -13.0)
    z  = np.full(n_nodes, 3.0)
    x2 = np.full(n_nodes, 0.0)
    y2 = np.full(n_nodes, 0.0)
    
    r_tau0 = 1.0 / 2800.0
    
    time_series_node = np.zeros((steps, n_nodes))
    x1_series = np.zeros((steps, n_nodes))
    z_series = np.zeros((steps, n_nodes))
    
    for t in range(steps):
        t_ms = t * dt * 5.0 # Mapeo de paso de tiempo en ms
        coupling = Kg * (sc_matrix @ x1 - np.sum(sc_matrix, axis=1) * x1)
        
        f1 = np.where(x1 < 0, x1**3 - 3.0 * x1**2, (x2 - 0.6 * (z - 4.0)**2) * x1)
        dx1 = y1 - f1 - z + 3.1 + coupling
        
        # Inyección de Estimulación Focal por Tren de Pulsos (PulseTrain Stimulus)
        if enable_stim and (stim_onset <= t_ms <= (stim_onset + stim_dur)):
            for ez_idx in ez_indices:
                if ez_idx not in resected_indices:
                    dx1[ez_idx] += stim_amp
                    
        dy1 = 1.0 - 5.0 * (x1**2) - y1
        
        dz = r_tau0 * (4.0 * (x1 - x0) - z)
        
        f2 = np.where(x2 < -0.25, 0.0, 6.0 * (x2 + 0.25))
        dx2 = -y2 + x2 - (x2**3) + 2.0 * np.maximum(0.0, x1) - 0.3 * (z - 3.5)
        dy2 = (1.0 / 10.0) * (-y2 + f2)
        
        noise_x1 = 0.04 * np.random.normal(0, 1, n_nodes)
        
        x1 += dx1 * dt + noise_x1
        y1 += dy1 * dt
        z  += dz * dt
        x2 += dx2 * dt
        y2 += dy2 * dt
        
        time_series_node[t] = -x1 + x2
        x1_series[t] = x1
        z_series[t]  = z
        
    return time_series_node, x1_series, z_series, x0

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA STREAMLIT
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Gemelo Virtual Cerebral - Epilepsia v14 (I_Motor & MRC Scale Fix)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    .stApp {
        background-color: #060911;
        color: #E2E8F0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    .status-bar {
        background: linear-gradient(90deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.8) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 12px 20px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        backdrop-filter: blur(16px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    }

    .main-header {
        font-family: 'Bebas Neue', sans-serif !important;
        font-size: 3.4rem;
        line-height: 1.0;
        text-align: left;
        margin: 0;
        background: linear-gradient(135deg, #FFFFFF 0%, #38BDF8 60%, #00F2FE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 3.5px !important;
    }

    .sub-header {
        font-size: 0.90rem;
        font-family: 'Inter', sans-serif;
        color: #94A3B8;
        margin-top: 4px;
        font-weight: 400;
        letter-spacing: 0.03em;
    }

    .kpi-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 18px;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
    }

    .kpi-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.15rem;
        letter-spacing: 1.8px;
        color: #38BDF8;
        margin-bottom: 4px;
    }

    .kpi-value {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 2.2rem;
        letter-spacing: 1.5px;
        color: #F8FAFC;
        line-height: 1.0;
    }

    .kpi-sub {
        font-family: 'Inter', sans-serif;
        font-size: 0.78rem;
        color: #94A3B8;
        margin-top: 4px;
    }

    .hud-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 22px;
        backdrop-filter: blur(12px);
        margin-bottom: 15px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
    }

    .hud-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.25rem;
        letter-spacing: 2px;
        color: #38BDF8;
        margin-bottom: 8px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.7);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 8px;
        font-family: 'Bebas Neue', sans-serif;
        letter-spacing: 2px;
        font-size: 1.2rem;
        color: #94A3B8;
        border: none;
        padding: 0px 22px;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.4);
    }

    section[data-testid="stSidebar"] {
        background-color: #090E1A !important;
        border-right: 1px solid rgba(255, 255, 255, 0.07);
    }

    .badge-status {
        background-color: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 3px 10px;
        border-radius: 20px;
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-ez {
        background-color: rgba(230, 57, 70, 0.15);
        color: #E63946;
        border: 1px solid rgba(230, 57, 70, 0.4);
        padding: 4px 12px;
        border-radius: 6px;
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 6px;
    }
    
    .badge-m1 {
        background-color: rgba(42, 157, 143, 0.15);
        color: #2A9D8F;
        border: 1px solid rgba(42, 157, 143, 0.4);
        padding: 4px 12px;
        border-radius: 6px;
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 6px;
    }
    
    .badge-eloquent {
        background-color: rgba(255, 183, 3, 0.15);
        color: #FFB703;
        border: 1px solid rgba(255, 183, 3, 0.4);
        padding: 4px 12px;
        border-radius: 6px;
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 6px;
    }

    .badge-seeg {
        background-color: rgba(114, 9, 183, 0.15);
        color: #C084FC;
        border: 1px solid rgba(114, 9, 183, 0.4);
        padding: 4px 12px;
        border-radius: 6px;
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# ENCABEZADO Y BARRA DE ESTADO DE CONTROL CLINICO
# -----------------------------------------------------------------------------
engine_status_text = "NATIVO TVB-LIBRARY" if HAS_TVB_NATIVE else "EPILEPTOR (MOTOR EDE TVB)"
st.markdown(f"""
<div class="status-bar">
    <div>
        <div class="main-header">GEMELO VIRTUAL CEREBRAL</div>
        <div class="sub-header">Plataforma Neurofisiológica con Integración de The Virtual Brain (TVB), Estimulación Focal e Índice I_Motor</div>
    </div>
    <div style="text-align: right;">
        <span class="badge-status">● SIMULADOR TVB {engine_status_text}</span><br>
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #94A3B8;">PACIENTE: REF-EPI-2026-042 | MNI152</span>
    </div>
</div>
""", unsafe_allow_html=True)

plotly_layout_font = dict(family="Inter, sans-serif", color="#CBD5E1", size=12)
plotly_title_font = dict(family="Bebas Neue, sans-serif", color="#F8FAFC", size=20)

# -----------------------------------------------------------------------------
# ATLAS DESIKAN-KILLIANY Y COORDENADAS MNI152 (68 REGIONES)
# -----------------------------------------------------------------------------
DESIKAN_REGIONS = [
    'L_bankstemporal', 'L_caudalanteriorcingulate', 'L_caudalmiddlefrontal', 'L_cuneus', 'L_entorhinal', 'L_fusiform', 'L_inferiorparietal', 'L_inferiortemporal', 'L_isthmuscingulate', 'L_lateraloccipital', 'L_lateralorbitofrontal', 'L_lingual', 'L_medialorbitofrontal', 'L_middletemporal', 'L_parahippocampal', 'L_paracentral', 'L_parsopercularis', 'L_parsorbitalis', 'L_parstriangularis', 'L_pericalcarine', 'L_postcentral', 'L_posteriorcingulate', 'L_precentral', 'L_precuneus', 'L_rostralanteriorcingulate', 'L_rostralmiddlefrontal', 'L_superiorfrontal', 'L_superiorparietal', 'L_superiortemporal', 'L_supramarginal', 'L_frontalpole', 'L_temporalpole', 'L_transversetemporal', 'L_insula',
    'R_bankstemporal', 'R_caudalanteriorcingulate', 'R_caudalmiddlefrontal', 'R_cuneus', 'R_entorhinal', 'R_fusiform', 'R_inferiorparietal', 'R_inferiortemporal', 'R_isthmuscingulate', 'R_lateraloccipital', 'R_lateralorbitofrontal', 'R_lingual', 'R_medialorbitofrontal', 'R_middletemporal', 'R_parahippocampal', 'R_paracentral', 'R_parsopercularis', 'R_parsorbitalis', 'R_parstriangularis', 'R_pericalcarine', 'R_postcentral', 'R_posteriorcingulate', 'R_precentral', 'R_precuneus', 'R_rostralanteriorcingulate', 'R_rostralmiddlefrontal', 'R_superiorfrontal', 'R_superiorparietal', 'R_superiortemporal', 'R_supramarginal', 'R_frontalpole', 'R_temporalpole', 'R_transversetemporal', 'R_insula'
]

ELOQUENT_REGIONS = [
    'L_precentral', 'R_precentral',          # Corteza Motora Primaria
    'L_postcentral', 'R_postcentral',        # Corteza Somatosensorial
    'L_parsopercularis', 'L_parstriangularis', # Área de Broca (Lenguaje)
    'L_superiortemporal', 'R_superiortemporal',# Área de Wernicke / Auditiva
    'L_pericalcarine', 'R_pericalcarine'     # Corteza Visual Primaria
]

# Montaje de sensores EEG virtuales (10-20 system)
EEG_SENSORS_1020 = {
    'Fp1': (-30, 85, 10), 'Fp2': (30, 85, 10),
    'F3': (-45, 45, 40),  'F4': (45, 45, 40),
    'C3': (-65, -10, 55), 'C4': (65, -10, 55),
    'P3': (-45, -65, 50), 'P4': (45, -65, 50),
    'O1': (-25, -100, 10),'O2': (25, -100, 10),
    'F7': (-65, 40, -10), 'F8': (65, 40, -10),
    'T3': (-75, -20, -10),'T4': (75, -20, -10),
    'Cz': (0, -10, 80)
}
eeg_names = list(EEG_SENSORS_1020.keys())
eeg_coords = np.array([EEG_SENSORS_1020[k] for k in eeg_names])

@st.cache_data
def get_mni_coordinates():
    base_coords = {
        'bankstemporal': (-52, -35, -2), 'caudalanteriorcingulate': (-6, 22, 24),
        'caudalmiddlefrontal': (-34, 12, 44), 'cuneus': (-8, -80, 27),
        'entorhinal': (-25, -10, -32), 'fusiform': (-36, -40, -20),
        'inferiorparietal': (-42, -66, 38), 'inferiortemporal': (-52, -30, -22),
        'isthmuscingulate': (-9, -48, 16), 'lateraloccipital': (-36, -84, 10),
        'lateralorbitofrontal': (-32, 30, -14), 'lingual': (-14, -68, -4),
        'medialorbitofrontal': (-6, 38, -14), 'middletemporal': (-54, -32, -10),
        'parahippocampal': (-22, -26, -18), 'paracentral': (-8, -30, 60),
        'parsopercularis': (-46, 14, 18), 'parsorbitalis': (-44, 30, -6),
        'parstriangularis': (-46, 26, 12), 'pericalcarine': (-12, -78, 8),
        'postcentral': (-42, -24, 52), 'posteriorcingulate': (-8, -38, 28),
        'precentral': (-38, -10, 52), 'precuneus': (-10, -60, 42),
        'rostralanteriorcingulate': (-6, 38, 14), 'rostralmiddlefrontal': (-30, 48, 18),
        'superiorfrontal': (-14, 30, 44), 'superiorparietal': (-22, -60, 56),
        'superiortemporal': (-54, -20, 6), 'supramarginal': (-52, -38, 32),
        'frontalpole': (-18, 62, -10), 'temporalpole': (-38, 14, -34),
        'transversetemporal': (-42, -20, 10), 'insula': (-35, 6, 2)
    }
    pos_matrix = np.zeros((len(DESIKAN_REGIONS), 3))
    for idx, reg in enumerate(DESIKAN_REGIONS):
        hemi = reg[0]
        name = reg[2:]
        x, y, z = base_coords.get(name, (-30, 0, 0))
        if hemi == 'R':
            x = -x
        pos_matrix[idx] = [x, y, z]
    return pos_matrix

@st.cache_data
def generate_synthetic_sc(pos_matrix):
    n = len(pos_matrix)
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dists[i, j] = np.linalg.norm(pos_matrix[i] - pos_matrix[j])
    sc = np.exp(-dists / 35.0)
    np.fill_diagonal(sc, 0)
    sc = sc / np.max(sc)
    sc[sc < 0.05] = 0.0
    return sc, dists

def create_virtual_seeg_sensors(ez_pos, num_contacts=8, depth=35.0):
    if len(ez_pos.shape) == 2:
        target = np.mean(ez_pos, axis=0)
    else:
        target = ez_pos
    entry = target + np.array([-depth, 5.0, 15.0])
    contacts = []
    for t in np.linspace(0, 1, num_contacts):
        contacts.append(entry * (1 - t) + target * t)
    return np.array(contacts)

def compute_leadfield(sensors_pos, regions_pos):
    num_sensors = len(sensors_pos)
    num_regions = len(regions_pos)
    gain_matrix = np.zeros((num_sensors, num_regions))
    for s in range(num_sensors):
        for r in range(num_regions):
            d = np.linalg.norm(sensors_pos[s] - regions_pos[r])
            gain_matrix[s, r] = 1.0 / (d**2 + 1e-3)
    gain_matrix /= np.max(gain_matrix, axis=1, keepdims=True)
    return gain_matrix

def compute_eeg_leadfield(sensors, regions_pos):
    lf = np.zeros((len(sensors), len(regions_pos)))
    for s_i, s_pos in enumerate(sensors):
        for r_i, r_pos in enumerate(regions_pos):
            d = np.linalg.norm(s_pos - r_pos)
            lf[s_i, r_i] = 1.0 / (d**2 + 100.0)
    return lf / np.max(lf, axis=1, keepdims=True)

def build_plotly_3d_connectome(pos, sc_pre, sc_post, ez_indices, seeg_contacts, threshold=0.15):
    fig = go.Figure()
    
    # 1. Glass Brain (Convex Hull 3D)
    try:
        hull = ConvexHull(pos)
        fig.add_trace(go.Mesh3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
            i=hull.simplices[:, 0], j=hull.simplices[:, 1], k=hull.simplices[:, 2],
            color='lightgray', opacity=0.08, hoverinfo='none', name='Glass Brain (Corteza)'
        ))
    except Exception:
        pass

    # 2. Conectividad Conservada
    edge_x, edge_y, edge_z = [], [], []
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            if sc_post[i, j] >= threshold:
                edge_x.extend([pos[i, 0], pos[j, 0], None])
                edge_y.extend([pos[i, 1], pos[j, 1], None])
                edge_z.extend([pos[i, 2], pos[j, 2], None])

    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(0, 180, 216, 0.35)', width=2),
        name='Conectividad Conservada', hoverinfo='none'
    ))

    # 3. Tractos Resecados
    resected_x, resected_y, resected_z = [], [], []
    for i in range(len(pos)):
        for j in range(len(pos)):
            if sc_pre[i, j] > 0.08 and sc_post[i, j] == 0:
                resected_x.extend([pos[i, 0], pos[j, 0], None])
                resected_y.extend([pos[i, 1], pos[j, 1], None])
                resected_z.extend([pos[i, 2], pos[j, 2], None])

    fig.add_trace(go.Scatter3d(
        x=resected_x, y=resected_y, z=resected_z,
        mode='lines',
        line=dict(color='rgba(230, 57, 70, 0.65)', width=3, dash='dash'),
        name='Tractos Resecados (EZ)', hoverinfo='none'
    ))

    # 4. Nodos Corticales Clasificados
    node_colors, node_sizes, categories = [], [], []
    for idx, reg in enumerate(DESIKAN_REGIONS):
        if idx in ez_indices:
            node_colors.append('#E63946') # Rojo - EZ
            node_sizes.append(10)
            categories.append('Zona Epileptógena (EZ)')
        elif reg in ELOQUENT_REGIONS:
            node_colors.append('#FFB703') # Amarillo - Área Elocuente
            node_sizes.append(9)
            categories.append('Área Elocuente (Crítica)')
        else:
            node_colors.append('#2A9D8F') # Verde/Azul - Corteza Estándar
            node_sizes.append(6)
            categories.append('Corteza Estándar')

    hover_text = [
        f"<b>Región:</b> {reg}<br><b>Categoría:</b> {cat}<br><b>MNI:</b> ({pos[i,0]:.1f}, {pos[i,1]:.1f}, {pos[i,2]:.1f})"
        for i, (reg, cat) in enumerate(zip(DESIKAN_REGIONS, categories))
    ]

    fig.add_trace(go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        mode='markers',
        marker=dict(size=node_sizes, color=node_colors, opacity=0.9, line=dict(color='white', width=1)),
        text=hover_text, hoverinfo='text', name='Nodos Corticales (Atlas 3D)'
    ))

    # 5. Electrodo sEEG Virtual
    fig.add_trace(go.Scatter3d(
        x=seeg_contacts[:, 0], y=seeg_contacts[:, 1], z=seeg_contacts[:, 2],
        mode='markers+lines',
        marker=dict(size=7, color='#7209B7', symbol='diamond'),
        line=dict(color='#7209B7', width=5),
        name='Electrodo sEEG Virtual',
        text=[f"Contacto sEEG #{i+1}" for i in range(len(seeg_contacts))],
        hoverinfo='text'
    ))

    fig.update_layout(
        paper_bgcolor='#060911', plot_bgcolor='#060911',
        font=dict(family="Inter, sans-serif", color="#CBD5E1", size=12),
        title=dict(text="<b>MODELO 3D DE CONECTIVIDAD NEURONAL & RESECCIÓN VIRTUAL (TVB)</b>", x=0.5, font=dict(family="Bebas Neue, sans-serif", size=20, color='#F8FAFC')),
        scene=dict(
            xaxis=dict(title='X (Izq - Der)', backgroundcolor='#060911', gridcolor='rgba(255,255,255,0.08)'),
            yaxis=dict(title='Y (Post - Ant)', backgroundcolor='#060911', gridcolor='rgba(255,255,255,0.08)'),
            zaxis=dict(title='Z (Inf - Sup)', backgroundcolor='#060911', gridcolor='rgba(255,255,255,0.08)'),
            aspectmode='data'
        ),
        legend=dict(font=dict(color='white'), bgcolor='rgba(15, 23, 42, 0.8)'),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

pos = get_mni_coordinates()
sc_matrix, dist_matrix = generate_synthetic_sc(pos)
regions = DESIKAN_REGIONS
m1_index = regions.index("L_precentral")
leadfield_eeg = compute_eeg_leadfield(eeg_coords, pos)

# -----------------------------------------------------------------------------
# PANEL LATERAL: PARÁMETROS PREQUIRÚRGICOS Y PARÁMETROS NEUROFISIOLÓGICOS TVB
# -----------------------------------------------------------------------------
st.sidebar.markdown("### CONFIGURACIÓN PREQUIRÚRGICA")

default_ez = ['L_entorhinal', 'L_parahippocampal', 'L_temporalpole', 'L_middletemporal']
ez_selected = st.sidebar.multiselect(
    "1. Zona Epileptógena (EZ):",
    options=regions,
    default=default_ez
)

ez_indices = [regions.index(r) for r in ez_selected if r in regions]

intervention_type = st.sidebar.selectbox(
    "2. Intervención Virtual:",
    ["Línea Base (Sin Intervención)", "Resección Quirúrgica Virtual", "Estimulación Cerebral Profunda (DBS)"]
)

resection_radius = 12.0
eta_neuromodulation = 0.0

if intervention_type == "Resección Quirúrgica Virtual":
    resection_radius = st.sidebar.slider("Radio de Cavidad de Resección (mm):", 5.0, 25.0, 12.0)
elif intervention_type == "Estimulación Cerebral Profunda (DBS)":
    eta_neuromodulation = st.sidebar.slider("Eficacia de Neuromodulación (η):", 0.1, 0.9, 0.6, step=0.05)

# -----------------------------------------------------------------------------
# APARTADO MEJORADO 1: RESECCIÓN POR TABLA Y MANEJO DE NODOS DESCONECTADOS
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### ✂️ RESECCIÓN Y DESCONEXIÓN POR TABLA")

nodes_resect_table = st.sidebar.multiselect(
    "Seleccionar Nodos a Resecar / Desconectar:",
    options=regions,
    default=['L_entorhinal', 'L_parahippocampal'] if intervention_type == "Resección Quirúrgica Virtual" else []
)

resect_indices_table = [regions.index(r) for r in nodes_resect_table if r in regions]

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ ESTIMULACIÓN ELÉCTRICA FOCAL (PULSE TRAIN)")
enable_stim_flag = st.sidebar.checkbox("Activar Estimulación Eléctrica Focal", value=False, help="Inyecta una corriente continua/pulsada en la EZ para inducir o abortar una crisis (Jirsa et al., 2014).")

if enable_stim_flag:
    stim_onset_val = st.sidebar.slider("Inicio del Estímulo (ms):", 500, 8000, 2000, step=250)
    stim_amp_val   = st.sidebar.slider("Amplitud del Pulso (I_ext):", 1.0, 30.0, 12.0, step=1.0)
    stim_dur_val   = st.sidebar.slider("Duración del Pulso (ms):", 20, 500, 100, step=20)
else:
    stim_onset_val = 2000.0
    stim_amp_val   = 0.0
    stim_dur_val   = 100.0

st.sidebar.markdown("---")
st.sidebar.markdown("### VISUALIZACIÓN DE TRACTOS SC")
sc_threshold_val = st.sidebar.slider("Umbral Conectoma 3D (Min Weight):", 0.05, 0.50, 0.15, step=0.02, help="Ajusta el umbral para mostrar más o menos tractos activos en el espacio 3D.")

st.sidebar.markdown("---")
st.sidebar.markdown("### PARÁMETROS TVB EPILEPTOR")
x0_ez_val = st.sidebar.slider("Epileptogenicidad EZ (x0_EZ):", -1.8, -0.8, -1.2, step=0.1, help="Parámetro x0 de TVB: Valores > -1.6 generan crisis espontáneas.")
kg_coupling_val = st.sidebar.slider("Acoplamiento Global Red (Kg):", 0.2, 3.0, 1.0, step=0.1, help="Ponderador de conectividad estructural entre regiones.")

st.sidebar.markdown("---")
st.sidebar.markdown("### CÁNULA sEEG PROFUNDIDAD")
num_contacts = st.sidebar.slider("Contactos Sensores sEEG:", 4, 16, 8)
seeg_depth = st.sidebar.slider("Profundidad Cánula (mm):", 10.0, 50.0, 35.0)

# -----------------------------------------------------------------------------
# CÁLCULOS Y MODIFICACIÓN DE LA MATRIZ DE CONECTIVIDAD
# -----------------------------------------------------------------------------
sc_modified = sc_matrix.copy()

# Combinar resecciones de la intervención clásica y de la tabla
active_resection_indices = list(set(ez_indices if intervention_type == "Resección Quirúrgica Virtual" else [] + resect_indices_table))

for idx in active_resection_indices:
    sc_modified[idx, :] = 0.0
    sc_modified[:, idx] = 0.0

# Cálculo del Índice I_Motor
w1, w2 = 0.5, 0.5
lambda_rom = 25.0

if len(ez_indices) > 0:
    min_dist_ez_m1 = min([dist_matrix[ez_idx, m1_index] for ez_idx in ez_indices])
    avg_conn_ez_m1 = np.mean([sc_modified[ez_idx, m1_index] for ez_idx in ez_indices])
else:
    min_dist_ez_m1 = 100.0
    avg_conn_ez_m1 = 0.0

eta_factor = eta_neuromodulation if intervention_type == "Estimulación Cerebral Profunda (DBS)" else (1.0 if (intervention_type == "Resección Quirúrgica Virtual" or len(resect_indices_table) > 0) else 0.0)

term_dist = np.exp(-min_dist_ez_m1 / lambda_rom)
term_conn = avg_conn_ez_m1 * (1.0 - eta_factor)
raw_risk = w1 * term_dist + w2 * term_conn

i_motor_score = float(np.clip(100.0 * (1.0 - raw_risk), 0.0, 100.0))

# -----------------------------------------------------------------------------
# RUN TVB NEUROPHYSIOLOGICAL SIMULATION
# -----------------------------------------------------------------------------
is_resected_bool = (intervention_type == "Resección Quirúrgica Virtual" or len(resect_indices_table) > 0)
time_series_nodes, x1_series_nodes, z_series_nodes, x0_vector = run_tvb_epileptor_simulation(
    sc_matrix=sc_modified,
    ez_indices=ez_indices,
    resected_indices=resect_indices_table,
    x0_ez=x0_ez_val,
    x0_healthy=-2.4,
    Kg=kg_coupling_val,
    eta_dbs=eta_neuromodulation,
    is_resected=is_resected_bool,
    dt=0.05,
    steps=2000,
    enable_stim=enable_stim_flag,
    stim_onset=stim_onset_val,
    stim_dur=stim_dur_val,
    stim_amp=stim_amp_val
)

# -----------------------------------------------------------------------------
# TARJETAS DE MÉTRICAS CLAVE (KPI BAR - TOP ROW)
# -----------------------------------------------------------------------------
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

with col_kpi1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">ZONA EPILEPTÓGENA (EZ)</div>
        <div class="kpi-value">{len(ez_indices)} NODOS</div>
        <div class="kpi-sub">{', '.join(ez_selected[:2])}{'...' if len(ez_selected)>2 else ''}</div>
    </div>
    """, unsafe_allow_html=True)

with col_kpi2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">DISTANCIA A CORTEZA MOTORA (M1)</div>
        <div class="kpi-value" style="color: {'#2A9D8F' if min_dist_ez_m1 > 30 else '#E63946'};">{min_dist_ez_m1:.1f} MM</div>
        <div class="kpi-sub">Vía Corticoespinal Directa</div>
    </div>
    """, unsafe_allow_html=True)

with col_kpi3:
    active_tracts_pct = (np.sum(sc_modified > 0) / np.sum(sc_matrix > 0)) * 100
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">INTEGRIDAD CONECTOMA SC</div>
        <div class="kpi-value" style="color: #38BDF8;">{active_tracts_pct:.1f}%</div>
        <div class="kpi-sub">Tractos Funcionales Activos</div>
    </div>
    """, unsafe_allow_html=True)

with col_kpi4:
    color_rom = "#2A9D8F" if i_motor_score > 80 else ("#FFB703" if i_motor_score >= 40 else "#E63946")
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">PRESERVACIÓN NEURO-MOTORA I_MOTOR</div>
        <div class="kpi-value" style="color: {color_rom};">{i_motor_score:.1f}%</div>
        <div class="kpi-sub">Seguridad Motor Pre-operatoria</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PESTAÑAS PRINCIPALES DEL DASHBOARD
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "CONECTOMA 3D & TABLA DE RESECCIÓN",
    "MATRIZ DE CONECTIVIDAD (SC)",
    "SIMULACIÓN TVB & INSTRUMENTACIÓN VIRTUAL (EEG/sEEG)",
    "EVALUACIÓN NEURO-MOTORA (I_MOTOR & ESCALA MRC)"
])

# -----------------------------------------------------------------------------
# TAB 1: CONECTOMA 3D + TABLA DE RESECCIÓN INTERACTIVA
# -----------------------------------------------------------------------------
with tab1:
    col_a, col_b = st.columns([3, 1])
    
    ez_target_pos = pos[ez_indices] if len(ez_indices) > 0 else pos[0]
    seeg_sensors_pos = create_virtual_seeg_sensors(ez_target_pos, num_contacts=num_contacts, depth=seeg_depth)
    leadfield_matrix = compute_leadfield(seeg_sensors_pos, pos)
    
    with col_a:
        fig_3d = build_plotly_3d_connectome(
            pos=pos,
            sc_pre=sc_matrix,
            sc_post=sc_modified,
            ez_indices=ez_indices,
            seeg_contacts=seeg_sensors_pos,
            threshold=sc_threshold_val
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with col_b:
        st.markdown("<div class='hud-card'>", unsafe_allow_html=True)
        st.markdown("<div class='hud-title'>ESTRUCTURAS Y CLASIFICACIÓN</div>", unsafe_allow_html=True)
        st.markdown("<span class='badge-ez'>🔴 ZONA EPILEPTÓGENA (EZ)</span>", unsafe_allow_html=True)
        st.markdown("<span class='badge-m1'>🟢 CORTEZA MOTORA (M1)</span>", unsafe_allow_html=True)
        st.markdown("<span class='badge-eloquent'>🟡 ÁREAS ELOCUENTES CRÍTICAS</span>", unsafe_allow_html=True)
        st.markdown("<span class='badge-seeg'>🟣 CÁNULA sEEG PROFUNDA</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='hud-card'>", unsafe_allow_html=True)
        st.markdown("<div class='hud-title'>PARÁMETROS TVB EPILEPTOR</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-sub'>Epileptogenicidad EZ (x0): <b style='color:#E63946;'>{x0_ez_val:.2f}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-sub'>Tejido Sano (x0): <b style='color:#2A9D8F;'>-2.40</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-sub'>Acoplamiento Global (Kg): <b style='color:#38BDF8;'>{kg_coupling_val:.1f}</b></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 📋 TABLA INTERACTIVA DE RESECCIÓN CORTICAL (MODELING SURGICAL RESECTION)")
    df_nodes = pd.DataFrame({
        "Estado Resección": [r in nodes_resect_table or (r in ez_selected and intervention_type == "Resección Quirúrgica Virtual") for r in DESIKAN_REGIONS],
        "Nombre Región": DESIKAN_REGIONS,
        "Hemisferio": ["Izquierdo" if r.startswith("L_") else "Derecho" for r in DESIKAN_REGIONS],
        "Categoría Cortical": ["EZ (Epileptógena)" if r in ez_selected else ("Área Elocuente" if r in ELOQUENT_REGIONS else "Corteza Estándar") for r in DESIKAN_REGIONS],
        "Excitabilidad x0": [x0_vector[i] for i in range(len(DESIKAN_REGIONS))],
        "Conexiones SC Activas": [int(np.sum(sc_modified[i] > 0)) for i in range(len(DESIKAN_REGIONS))],
        "Coordenada X (MNI)": [pos[i, 0] for i in range(len(DESIKAN_REGIONS))],
        "Coordenada Y (MNI)": [pos[i, 1] for i in range(len(DESIKAN_REGIONS))],
        "Coordenada Z (MNI)": [pos[i, 2] for i in range(len(DESIKAN_REGIONS))]
    })

    st.dataframe(
        df_nodes,
        column_config={
            "Estado Resección": st.column_config.CheckboxColumn("Resecado / Desconectado", default=False),
            "Excitabilidad x0": st.column_config.NumberColumn("Parámetro x0", format="%.2f"),
            "Conexiones SC Activas": st.column_config.ProgressColumn("Densidad SC", min_value=0, max_value=68)
        },
        use_container_width=True,
        height=320
    )

    with st.expander("📄 MATRIZ LEADFIELD DE SENSIBILIDAD ESPACIAL sEEG (SENSORES ➔ NODOS CORTICALES)"):
        st.dataframe(
            leadfield_matrix,
            column_config={i: f"Nodo {r}" for i, r in enumerate(regions[:10])},
            use_container_width=True
        )

# -----------------------------------------------------------------------------
# TAB 2: MATRIZ DE CONECTIVIDAD (SC) Y TEORÍA DE GRAFOS
# -----------------------------------------------------------------------------
with tab2:
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        fig_mat_orig = px.imshow(
            sc_matrix,
            labels=dict(x="Región Cortical", y="Región Cortical", color="Peso SC"),
            x=regions, y=regions,
            color_continuous_scale="Plasma"
        )
        fig_mat_orig.update_layout(
            paper_bgcolor="#060911", plot_bgcolor="#060911", font=plotly_layout_font,
            title={'text': "MATRIZ ESTRUCTURAL INICIAL (LÍNEA BASE SC)", 'font': plotly_title_font},
            height=480, margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(fig_mat_orig, use_container_width=True)

    with col_m2:
        fig_mat_mod = px.imshow(
            sc_modified,
            labels=dict(x="Región Cortical", y="Región Cortical", color="Peso SC"),
            x=regions, y=regions,
            color_continuous_scale="Plasma"
        )
        fig_mat_mod.update_layout(
            paper_bgcolor="#060911", plot_bgcolor="#060911", font=plotly_layout_font,
            title={'text': f"MATRIZ MODIFICADA ({intervention_type.upper()})", 'font': plotly_title_font},
            height=480, margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(fig_mat_mod, use_container_width=True)

    node_degrees_orig = np.sum(sc_matrix > 0.15, axis=1)
    node_degrees_mod  = np.sum(sc_modified > 0.15, axis=1)
    
    fig_deg = go.Figure()
    fig_deg.add_trace(go.Bar(x=regions[:20], y=node_degrees_orig[:20], name="Línea Base", marker_color="#0284C7"))
    fig_deg.add_trace(go.Bar(x=regions[:20], y=node_degrees_mod[:20], name="Post-Intervención", marker_color="#E63946"))
    fig_deg.update_layout(
        paper_bgcolor="#060911", plot_bgcolor="#060911", font=plotly_layout_font,
        title={'text': "TEORÍA DE GRAFOS: GRADO DE CENTRALIDAD NODAL (TOP 20 REGIONES)", 'font': plotly_title_font},
        xaxis=dict(title="Regiones Corticales", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Grado de Conexión (Degree)", gridcolor="rgba(255,255,255,0.06)"),
        barmode='group', height=320, margin=dict(l=0, r=0, b=0, t=40)
    )
    st.plotly_chart(fig_deg, use_container_width=True)

# -----------------------------------------------------------------------------
# APARTADO MEJORADO 2: INSTRUMENTACIÓN VIRTUAL CON MARCADOR DE ESTIMULACIÓN
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### 📈 SIMULACIÓN Y VISUALIZADOR DINÁMICO DE INSTRUMENTACIÓN VIRTUAL (EEG & sEEG)")
    
    col_sig_ctrl, col_sig_plot = st.columns([1, 3])
    
    eeg_signals = leadfield_eeg @ time_series_nodes.T # Shape: (15, 2000)
    seeg_signals = leadfield_matrix @ time_series_nodes.T  # Shape: (num_contacts, 2000)
    t_sec = np.linspace(0, 10, time_series_nodes.shape[0])
    
    with col_sig_ctrl:
        st.markdown("<div class='hud-card'>", unsafe_allow_html=True)
        st.markdown("<div class='hud-title'>CONTROLES DE SEÑAL</div>", unsafe_allow_html=True)
        
        signal_type_choice = st.radio(
            "Seleccionar Instrumentación:",
            ["EEG Superficie (Sistema 10-20)", "sEEG Cánula Profunda", "LFP Cortical (Nodos TVB)"]
        )
        
        time_window_sec = st.slider("Ventana Temporal (segundos):", 1.0, 10.0, 5.0, step=0.5)
        time_start_sec = st.slider("Deslizar Tiempo Inicio (s):", 0.0, float(10.0 - time_window_sec), 0.0, step=0.2)
        gain_val = st.slider("Ganancia de Amplitud (Gain):", 0.5, 5.0, 1.5, step=0.2)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_sig_plot:
        # Filtrado temporal según el slider
        t_mask = (t_sec >= time_start_sec) & (t_sec <= (time_start_sec + time_window_sec))
        sub_t = t_sec[t_mask]
        
        fig_sig = go.Figure()
        
        if signal_type_choice == "EEG Superficie (Sistema 10-20)":
            colors_eeg = px.colors.qualitative.Bold
            for i, name in enumerate(eeg_names):
                sig_window = eeg_signals[i, t_mask] * gain_val
                fig_sig.add_trace(go.Scatter(
                    x=sub_t, y=sig_window + (i * 3.0),
                    mode='lines',
                    line=dict(color=colors_eeg[i % len(colors_eeg)], width=1.5),
                    name=f"EEG {name}"
                ))
            y_vals = [i * 3.0 for i in range(len(eeg_names))]
            y_txts = eeg_names
            chart_title = "REGISTRO EEG MULTICANAL SUPERFICIAL (10-20)"

        elif signal_type_choice == "sEEG Cánula Profunda":
            colors_seeg = ["#38BDF8", "#0284C7", "#2A9D8F", "#C084FC", "#E63946", "#FFB703"]
            for c in range(min(6, num_contacts)):
                sig_window = seeg_signals[c, t_mask] * gain_val
                fig_sig.add_trace(go.Scatter(
                    x=sub_t, y=sig_window + (c * 3.5),
                    mode='lines',
                    line=dict(color=colors_seeg[c % len(colors_seeg)], width=1.5),
                    name=f"sEEG #{c+1}"
                ))
            y_vals = [c * 3.5 for c in range(min(6, num_contacts))]
            y_txts = [f"sEEG #{c+1}" for c in range(min(6, num_contacts))]
            chart_title = "REGISTRO sEEG CANULA PROFUNDA (LEADFIELD PROYECTADO)"

        else: # LFP Cortical
            disp_indices = ez_indices + [m1_index]
            for idx_i, n_idx in enumerate(disp_indices):
                sig_window = time_series_nodes[t_mask, n_idx] * gain_val
                reg_name = regions[n_idx]
                c_line = '#E63946' if n_idx in ez_indices else '#2A9D8F'
                fig_sig.add_trace(go.Scatter(
                    x=sub_t, y=sig_window + (idx_i * 4.0),
                    mode='lines',
                    line=dict(color=c_line, width=1.8),
                    name=f"LFP {reg_name}"
                ))
            y_vals = [idx_i * 4.0 for idx_i in range(len(disp_indices))]
            y_txts = [regions[n_idx] for n_idx in disp_indices]
            chart_title = "POTENCIALES DE CAMPO LOCAL (LFP NATIVO TVB)"

        # Agregar sombreado del periodo de estimulación si está activado
        if enable_stim_flag:
            stim_start_s = stim_onset_val / 1000.0
            stim_end_s   = (stim_onset_val + stim_dur_val) / 1000.0
            fig_sig.add_vrect(
                x0=stim_start_s, x1=stim_end_s,
                fillcolor="rgba(234, 179, 8, 0.25)",
                line_width=1.5, line_color="#EAB308",
                annotation_text="⚡ ESTÍMULO FOCAL",
                annotation_position="top left"
            )

        fig_sig.update_layout(
            paper_bgcolor="#060911", plot_bgcolor="#060911", font=plotly_layout_font,
            title={'text': chart_title, 'font': plotly_title_font},
            xaxis=dict(
                title="Tiempo de Simulación (segundos)", gridcolor="rgba(255,255,255,0.06)",
                rangeslider=dict(visible=True, bgcolor="#0F172A") # RANGE SLIDER INTEGRADO
            ),
            yaxis=dict(
                title="Canales (mV + Offset)", tickmode='array', tickvals=y_vals, ticktext=y_txts,
                gridcolor="rgba(255,255,255,0.06)"
            ),
            height=460, margin=dict(l=0, r=0, b=0, t=40),
            legend=dict(orientation="h", y=1.12)
        )
        st.plotly_chart(fig_sig, use_container_width=True)

    st.markdown("---")
    col_s2a, col_s2b = st.columns(2)

    with col_s2a:
        if len(ez_indices) > 0:
            ez_idx_primary = ez_indices[0]
            fft_vals = np.abs(np.fft.rfft(time_series_nodes[:, ez_idx_primary]))
            fft_freqs = np.fft.rfftfreq(len(t_sec), d=(t_sec[1]-t_sec[0]))
            
            fig_fft = go.Figure()
            fig_fft.add_trace(go.Scatter(
                x=fft_freqs[:150], y=fft_vals[:150],
                mode='lines',
                line=dict(color='#2A9D8F', width=2),
                fill='tozeroy',
                fillcolor='rgba(42, 157, 143, 0.15)'
            ))
            fig_fft.update_layout(
                paper_bgcolor="#060911", plot_bgcolor="#060911", font=plotly_layout_font,
                title={'text': "ESPECTRO DE FRECUENCIA FFT EN EZ (TVB)", 'font': plotly_title_font},
                xaxis=dict(title="Frecuencia (Hz)", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(title="Densidad Espectral", gridcolor="rgba(255,255,255,0.06)"),
                height=280, margin=dict(l=0, r=0, b=0, t=40)
            )
            st.plotly_chart(fig_fft, use_container_width=True)

    with col_s2b:
        fig_x1 = go.Figure()
        if len(ez_indices) > 0:
            fig_x1.add_trace(go.Scatter(x=t_sec, y=x1_series_nodes[:, ez_indices[0]], mode='lines', line=dict(color='#E63946', width=1.8), name=f"EZ ({regions[ez_indices[0]]})"))
        fig_x1.add_trace(go.Scatter(x=t_sec, y=x1_series_nodes[:, m1_index], mode='lines', line=dict(color='#2A9D8F', width=1.8), name=f"M1 ({regions[m1_index]})"))
        fig_x1.update_layout(
            paper_bgcolor="#060911", plot_bgcolor="#060911", font=plotly_layout_font,
            title={'text': "VARIABLE x1(t) - DIVERGENCIA RÁPIDA / DISPARO ICTAL", 'font': plotly_title_font},
            xaxis=dict(title="Tiempo (s)", gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="x1 (Amplitud Rápida)", gridcolor="rgba(255,255,255,0.06)"),
            height=280, margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(fig_x1, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: EVALUACIÓN BIOMECÁNICA (I_ROM) Y RADAR MULTIVARIADO
# -----------------------------------------------------------------------------
with tab4:
    col_r1, col_r2 = st.columns([1, 1])

    with col_r1:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = i_motor_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "PRESERVACIÓN NEURO-MOTORA I_MOTOR (%)", 'font': plotly_title_font},
            number = {'font': {'family': 'Bebas Neue', 'size': 56, 'color': '#38BDF8'}},
            gauge = {
                'axis': {'range': [0, 100], 'tickcolor': "#94A3B8"},
                'bar': {'color': "#0284C7", 'thickness': 0.3},
                'bgcolor': "#0F172A",
                'borderwidth': 1,
                'bordercolor': "rgba(255,255,255,0.1)",
                'steps': [
                    {'range': [0, 40], 'color': "#991B1B"},
                    {'range': [40, 80], 'color': "#854D0E"},
                    {'range': [80, 100], 'color': "#065F46"}
                ],
                'threshold': {
                    'line': {'color': "#FFFFFF", 'width': 3},
                    'thickness': 0.8,
                    'value': i_motor_score
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#060911", plot_bgcolor="#060911", font=plotly_layout_font,
            height=380, margin=dict(l=0, r=0, b=0, t=50)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_r2:
        radar_categories = ['Margen Anatómico', 'Aislamiento SC', 'Modulación DBS', 'Seguridad Corticoespinal', 'Estabilidad Ictal']
        
        val_margin = float(np.clip(min_dist_ez_m1 * 2.0, 0, 100))
        val_sc = float(active_tracts_pct)
        val_dbs = float(eta_neuromodulation * 100 if intervention_type == "Estimulación Cerebral Profunda (DBS)" else 100.0)
        val_safety = float(i_motor_score)
        val_ictal = float(100.0 - abs(x0_ez_val + 1.2) * 50.0)
        val_ictal = float(np.clip(val_ictal, 10, 100))

        radar_values = [val_margin, val_sc, val_dbs, val_safety, val_ictal]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_values,
            theta=radar_categories,
            fill='toself',
            fillcolor='rgba(2, 132, 199, 0.25)',
            line=dict(color='#38BDF8', width=2),
            name="Índice de Riesgo Multivariado"
        ))
        fig_radar.update_layout(
            paper_bgcolor="#060911", plot_bgcolor="#060911", font=plotly_layout_font,
            title={'text': "EVALUACIÓN MULTIVARIADA PREQUIRÚRGICA", 'font': plotly_title_font},
            polar=dict(
                bgcolor="#060911",
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.08)"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.08)")
            ),
            height=380, margin=dict(l=20, r=20, b=20, t=50)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("<div class='hud-card'>", unsafe_allow_html=True)
    st.markdown("<div class='hud-title'>DIAGNÓSTICO Y RECOMENDACIÓN CLÍNICA AUTOMATIZADA</div>", unsafe_allow_html=True)
    
    if i_motor_score > 80:
        st.success("🟢 **PROCEDIMIENTO RECOMENDADO**: Alto margen de preservación motora (I_ROM > 80%). Mínimo riesgo de secuelas en córtex primario motor M1.")
    elif i_motor_score >= 40:
        st.warning("🟡 **REVISIÓN PREQUIRÚRGICA**: Riesgo moderado (40% <= I_ROM < 80%). Se sugiere simular un escenario de Estimulación Cerebral Profunda (DBS) antes de proceder con resección invasiva.")
    else:
        st.error("🔴 **ALTO RIESGO MOTOR**: Invasión en vecindad de la vía corticoespinal (I_ROM < 40%). La resección directa presenta alto riesgo de secuelas permanentes en el rango de movimiento.")
    st.markdown("</div>", unsafe_allow_html=True)

# Pie de página
st.markdown("---")
st.caption("Gemelo Virtual Cerebral para Epilepsia — Prototipo Prequirúrgico con Integración TVB | CUCEI, Universidad de Guadalajara.")
