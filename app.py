import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import anthropic

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Simulador Cacao · Monte Carlo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
    }
    .stApp {
        background: #0d1117;
        color: #e6edf3;
    }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-card .label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #8b949e;
        font-family: 'Space Mono', monospace;
        margin-bottom: 8px;
    }
    .metric-card .value {
        font-size: 28px;
        font-weight: 800;
        font-family: 'Space Mono', monospace;
    }
    .metric-card.pesimista .value { color: #f85149; }
    .metric-card.mediana .value   { color: #58a6ff; }
    .metric-card.optimista .value { color: #3fb950; }
    .metric-card.var .value       { color: #d29922; }

    .info-box {
        background: #161b22;
        border-left: 3px solid #58a6ff;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        font-family: 'Space Mono', monospace;
        font-size: 12px;
        color: #8b949e;
        margin: 8px 0;
    }
    h1, h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; }
    .stSlider > div > div { background: #21262d; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ENCABEZADO
# ─────────────────────────────────────────────
st.markdown("## 📊 Simulador de Precios · Cacao Ecuatoriano")
st.markdown(
    "<span style='font-family:Space Mono,monospace;font-size:12px;color:#8b949e;'>"
    "500 trayectorias · GBM con fat tails · Percentiles 10/50/90 · VaR 95%</span>",
    unsafe_allow_html=True
)
st.divider()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Parámetros Comerciales")

descuento_tonelada = st.sidebar.number_input(
    "Descuento Logístico/Calidad (USD/MT):", value=500, step=10,
    help="Spread habitual entre ICE NY y precio FOB Ecuador."
)
breakeven = st.sidebar.number_input(
    "Precio de equilibrio del exportador (USD/MT):", value=6000, step=50,
    help="Tu costo total por tonelada. Se dibuja como línea de referencia en el gráfico."
)
unidad = st.sidebar.selectbox("Mostrar precio en:", ["Quintales (qq)", "Toneladas Métricas (MT)"])
dias_proyeccion = st.sidebar.slider("Días a proyectar:", min_value=7, max_value=30, value=14)

st.sidebar.divider()
st.sidebar.markdown("### 🎲 Reproducibilidad")
usar_semilla = st.sidebar.checkbox("Fijar semilla aleatoria", value=True,
    help="Activa esto para que cada recarga dé los mismos resultados.")
semilla = st.sidebar.number_input("Valor de semilla:", value=42, step=1,
    disabled=not usar_semilla)

st.sidebar.divider()
st.sidebar.markdown("### 📉 Modelo Estadístico")
modelo = st.sidebar.radio(
    "Distribución de shocks:",
    ["Normal (GBM clásico)", "t de Student (fat tails)"],
    index=1,
    help="La distribución t captura mejor los movimientos extremos históricos del cacao."
)
grados_libertad = 4
if modelo == "t de Student (fat tails)":
    grados_libertad = st.sidebar.slider(
        "Grados de libertad (ν):", min_value=2, max_value=30, value=4,
        help="Valores bajos (2–5) = colas más gruesas = más eventos extremos. "
             "A partir de ~30 se aproxima a la normal."
    )

# ─────────────────────────────────────────────
# DATOS HISTÓRICOS
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def obtener_datos_cacao():
    cacao = yf.Ticker("CC=F")
    df = cacao.history(period="1y")
    return df['Close']

try:
    with st.spinner("Obteniendo precios ICE Nueva York..."):
        precios_historicos = obtener_datos_cacao()

    precio_hoy = precios_historicos.iloc[-1]
    retornos_diarios = precios_historicos.pct_change().dropna()

    # Volatilidad e histórico a 6m y 1y para referencia
    vol_6m = retornos_diarios.iloc[-126:].std()
    vol_1y = retornos_diarios.std()
    drift   = retornos_diarios.mean()

    st.sidebar.divider()
    st.sidebar.markdown("### 📐 Volatilidad Calibrada")
    volatilidad_ajustada = st.sidebar.slider(
        "Volatilidad Diaria (%):",
        min_value=0.5, max_value=5.0,
        value=float(vol_6m * 100), step=0.1,
        help="Por defecto usa los últimos 6 meses."
    ) / 100.0

    st.sidebar.markdown(
        f"<div class='info-box'>"
        f"Vol 6m: <b>{vol_6m*100:.2f}%</b> | Vol 1y: <b>{vol_1y*100:.2f}%</b>"
        f"</div>",
        unsafe_allow_html=True
    )

    # ─────────────────────────────────────────
    # MOTOR MONTE CARLO
    # ─────────────────────────────────────────
    if usar_semilla:
        np.random.seed(int(semilla))

    num_simulaciones = 500

    if modelo == "Normal (GBM clásico)":
        shocks = np.random.normal(
            loc=drift, scale=volatilidad_ajustada,
            size=(dias_proyeccion, num_simulaciones)
        )
    else:
        # t de Student: escalar para que la std coincida con la volatilidad objetivo
        t_samples = np.random.standard_t(df=grados_libertad, size=(dias_proyeccion, num_simulaciones))
        escala = volatilidad_ajustada / np.sqrt(grados_libertad / (grados_libertad - 2))
        shocks = drift + t_samples * escala

    trayectorias_ice = precio_hoy * np.cumprod(1 + shocks, axis=0)

    # Ajuste Ecuador
    trayectorias_ecuador = trayectorias_ice - descuento_tonelada
    factor_qq = 0.04536
    be_graf = breakeven  # breakeven en MT para el gráfico

    if unidad == "Quintales (qq)":
        trayectorias_ecuador = trayectorias_ecuador * factor_qq
        be_graf = breakeven * factor_qq
        etiqueta_y = "USD / Quintal"
        precio_actual_graf = (precio_hoy - descuento_tonelada) * factor_qq
    else:
        etiqueta_y = "USD / Tonelada Métrica"
        precio_actual_graf = precio_hoy - descuento_tonelada

    # Percentiles
    p10 = np.percentile(trayectorias_ecuador, 10, axis=1)
    p50 = np.percentile(trayectorias_ecuador, 50, axis=1)
    p90 = np.percentile(trayectorias_ecuador, 90, axis=1)

    # VaR 95% — pérdida máxima esperada al final de la proyección
    precios_finales = trayectorias_ecuador[-1, :]
    var_95 = precio_actual_graf - np.percentile(precios_finales, 5)
    cvar_95 = precio_actual_graf - precios_finales[precios_finales <= np.percentile(precios_finales, 5)].mean()

    # Porcentaje de escenarios sobre breakeven
    pct_ganancia = (precios_finales >= be_graf).mean() * 100

    fechas_futuras = [
        (datetime.today() + timedelta(days=i + 1)).strftime('%d %b') for i in range(dias_proyeccion)
    ]

    # ─────────────────────────────────────────
    # MÉTRICAS SUPERIORES
    # ─────────────────────────────────────────
    col_precio, col_var, col_cvar, col_be = st.columns(4)

    with col_precio:
        st.markdown(
            f"<div class='metric-card mediana'>"
            f"<div class='label'>Precio Actual Ecuador</div>"
            f"<div class='value'>${precio_actual_graf:,.2f}</div>"
            f"</div>", unsafe_allow_html=True
        )
    with col_var:
        st.markdown(
            f"<div class='metric-card var'>"
            f"<div class='label'>VaR 95% · Día {dias_proyeccion}</div>"
            f"<div class='value'>-${var_95:,.2f}</div>"
            f"</div>", unsafe_allow_html=True
        )
    with col_cvar:
        st.markdown(
            f"<div class='metric-card pesimista'>"
            f"<div class='label'>CVaR 95% (Expected Shortfall)</div>"
            f"<div class='value'>-${cvar_95:,.2f}</div>"
            f"</div>", unsafe_allow_html=True
        )
    with col_be:
        color_be = "optimista" if pct_ganancia >= 50 else "pesimista"
        st.markdown(
            f"<div class='metric-card {color_be}'>"
            f"<div class='label'>Escenarios sobre Break-Even</div>"
            f"<div class='value'>{pct_ganancia:.1f}%</div>"
            f"</div>", unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # GRÁFICOS
    # ─────────────────────────────────────────
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.65, 0.35],
        subplot_titles=(
            f"Cono de Probabilidad · {dias_proyeccion} días",
            f"Distribución Precios · Día {dias_proyeccion}"
        )
    )

    # — Cono: banda de incertidumbre —
    fig.add_trace(go.Scatter(
        x=fechas_futuras + fechas_futuras[::-1],
        y=list(p90) + list(p10[::-1]),
        fill='toself',
        fillcolor='rgba(88, 166, 255, 0.08)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Rango 10–90%',
        showlegend=True
    ), row=1, col=1)

    # — Cono: líneas de percentiles —
    fig.add_trace(go.Scatter(
        x=fechas_futuras, y=p90,
        line=dict(color='#3fb950', width=2, dash='dot'),
        name='Optimista P90'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=fechas_futuras, y=p50,
        line=dict(color='#58a6ff', width=3),
        name='Mediana P50'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=fechas_futuras, y=p10,
        line=dict(color='#f85149', width=2, dash='dot'),
        name='Pesimista P10'
    ), row=1, col=1)

    # — Línea break-even —
    fig.add_hline(
        y=be_graf, line_color='#d29922', line_dash='dash', line_width=1.5,
        annotation_text=f"Break-Even ${be_graf:,.2f}",
        annotation_font_color='#d29922',
        annotation_position="bottom right",
        row=1, col=1
    )

    # — Histograma distribución final —
    fig.add_trace(go.Histogram(
        x=precios_finales,
        nbinsx=40,
        marker_color='#58a6ff',
        marker_line_color='#0d1117',
        marker_line_width=0.5,
        opacity=0.85,
        name=f'Día {dias_proyeccion}',
        showlegend=False
    ), row=1, col=2)

    # VaR en histograma
    var_threshold = np.percentile(precios_finales, 5)
    fig.add_vline(
        x=var_threshold, line_color='#f85149', line_dash='dash', line_width=2,
        annotation_text=f"VaR 95%<br>${var_threshold:,.2f}",
        annotation_font_color='#f85149',
        annotation_position="top left",
        row=1, col=2
    )
    fig.add_vline(
        x=p50[-1], line_color='#58a6ff', line_dash='dash', line_width=2,
        annotation_text=f"Mediana<br>${p50[-1]:,.2f}",
        annotation_font_color='#58a6ff',
        annotation_position="top right",
        row=1, col=2
    )
    fig.add_vline(
        x=be_graf, line_color='#d29922', line_dash='dash', line_width=1.5,
        annotation_text="BE",
        annotation_font_color='#d29922',
        row=1, col=2
    )

    fig.update_layout(
        height=480,
        paper_bgcolor='#0d1117',
        plot_bgcolor='#161b22',
        font=dict(family='Space Mono, monospace', color='#8b949e', size=11),
        legend=dict(
            bgcolor='#161b22', bordercolor='#30363d', borderwidth=1,
            font=dict(size=10)
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        title_font_color='#e6edf3',
    )
    fig.update_xaxes(gridcolor='#21262d', linecolor='#30363d', tickfont=dict(size=10))
    fig.update_yaxes(gridcolor='#21262d', linecolor='#30363d',
                     tickprefix='$', tickfont=dict(size=10))
    fig.layout.annotations[0].font.color = '#e6edf3'
    fig.layout.annotations[1].font.color = '#e6edf3'

    st.plotly_chart(fig, use_container_width=True)

    # ─────────────────────────────────────────
    # TARJETAS DE RESUMEN DÍA N
    # ─────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"<div class='metric-card pesimista'>"
            f"<div class='label'>Pesimista P10 · Día {dias_proyeccion}</div>"
            f"<div class='value'>${p10[-1]:,.2f}</div>"
            f"</div>", unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"<div class='metric-card mediana'>"
            f"<div class='label'>Mediana P50 · Día {dias_proyeccion}</div>"
            f"<div class='value'>${p50[-1]:,.2f}</div>"
            f"</div>", unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"<div class='metric-card optimista'>"
            f"<div class='label'>Optimista P90 · Día {dias_proyeccion}</div>"
            f"<div class='value'>${p90[-1]:,.2f}</div>"
            f"</div>", unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # TABLA DETALLADA
    # ─────────────────────────────────────────
    with st.expander("📋 Tabla de rangos de probabilidad diaria"):
        df_tabla = pd.DataFrame({
            "Pesimista (P10)": p10,
            "Mediana (P50)":   p50,
            "Optimista (P90)": p90,
            "Rango (P90–P10)": p90 - p10
        }, index=fechas_futuras)
        st.dataframe(
            df_tabla.style
                .format("${:,.2f}")
                .background_gradient(subset=["Rango (P90–P10)"], cmap="YlOrRd"),
            use_container_width=True
        )

    # ─────────────────────────────────────────
    # ANALISTA IA
    # ─────────────────────────────────────────
    # Variables de modelo (necesarias para el analista IA y las notas)
    modelo_str = f"t de Student (ν={grados_libertad})" if "Student" in modelo else "Normal"
    semilla_str = str(int(semilla)) if usar_semilla else "No fijada"

    st.divider()
    st.markdown("### 🤖 Análisis con Inteligencia Artificial")
    st.markdown(
        "<span style='font-family:Space Mono,monospace;font-size:12px;color:#8b949e;'>"
        "Claude analiza el cono de probabilidad y te da una recomendación comercial concreta.</span>",
        unsafe_allow_html=True
    )

    # Construir resumen del contexto para Claude
    unidad_label = "quintal (qq)" if "Quintal" in unidad else "tonelada métrica (MT)"
    tendencia_mediana = p50[-1] - p50[0]
    tendencia_str = f"+${tendencia_mediana:,.2f}" if tendencia_mediana >= 0 else f"-${abs(tendencia_mediana):,.2f}"
    escenario_be = "FAVORABLE" if pct_ganancia >= 60 else ("NEUTRAL" if pct_ganancia >= 40 else "DESFAVORABLE")

    contexto_simulacion = f"""
Eres un analista experto en mercados de cacao para exportadores ecuatorianos.
Analiza estos resultados de una simulación Monte Carlo ({num_simulaciones} escenarios,
modelo GBM con distribución {modelo_str}, volatilidad diaria {volatilidad_ajustada*100:.2f}%) y
da una recomendación comercial clara y práctica.

DATOS DE LA SIMULACIÓN (precios en USD por {unidad_label}):
- Precio actual en Ecuador (post-descuento): ${precio_actual_graf:,.2f}
- Horizonte proyectado: {dias_proyeccion} días
- Descuento logístico/calidad aplicado: ${descuento_tonelada} USD/MT
- Break-even del exportador: ${be_graf:,.2f} por {unidad_label}

PROYECCIÓN AL DÍA {dias_proyeccion}:
- Escenario pesimista (P10): ${p10[-1]:,.2f}
- Escenario mediana (P50):   ${p50[-1]:,.2f}
- Escenario optimista (P90): ${p90[-1]:,.2f}
- Tendencia de la mediana:   {tendencia_str} en el período
- Rango de incertidumbre final (P90-P10): ${p90[-1]-p10[-1]:,.2f}

MÉTRICAS DE RIESGO:
- VaR 95% al día {dias_proyeccion}: -${var_95:,.2f} por {unidad_label}
- CVaR 95% (Expected Shortfall): -${cvar_95:,.2f} por {unidad_label}
- % de escenarios sobre break-even: {pct_ganancia:.1f}% → Contexto: {escenario_be}
- Volatilidad histórica 6m: {vol_6m*100:.2f}% | 1y: {vol_1y*100:.2f}%

INSTRUCCIONES DE RESPUESTA:
Responde en español. Estructura tu análisis así (usa exactamente estos encabezados en negrita):

**📊 Lectura del Cono**
2-3 oraciones interpretando qué dice la forma del cono sobre la incertidumbre del mercado.

**⚖️ Balance Riesgo/Oportunidad**
Evalúa si el momento es favorable o no para el exportador, considerando el break-even y el VaR.

**🎯 Recomendación**
Una acción concreta: vender ahora, esperar N días, cubrir con forward, o diversificar entregas. Sé directo.

**⚠️ Factores a Vigilar**
2-3 factores específicos del mercado de cacao que podrían invalidar esta proyección.

Importante: no repitas los números exactos de la simulación más de lo necesario. Sé conciso, directo y útil para alguien que va a tomar una decisión comercial hoy.
"""

    if st.button("🔍 Generar análisis con IA", type="primary", use_container_width=True):
        with st.spinner("Claude está analizando el cono de probabilidad..."):
            try:
                client = anthropic.Anthropic()

                with client.messages.stream(
                    model="claude-sonnet-4-5",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": contexto_simulacion}]
                ) as stream:
                    respuesta_placeholder = st.empty()
                    texto_acumulado = ""

                    for texto in stream.text_stream:
                        texto_acumulado += texto
                        # Escapar $ sueltos para que Streamlit no los interprete como LaTeX
                        texto_render = texto_acumulado.replace("$", "\\$")
                        respuesta_placeholder.markdown(texto_render)

                st.markdown(
                    "<div class='info-box' style='margin-top:12px;'>⚠️ Este análisis es generado por IA "
                    "con fines informativos. No constituye asesoría financiera ni recomendación de inversión. "
                    "Consulta con tu agente comercial antes de tomar decisiones de cobertura.</div>",
                    unsafe_allow_html=True
                )

            except anthropic.AuthenticationError:
                st.error("❌ API key de Anthropic no encontrada o inválida.")
                st.info(
                    "Para usar el analista IA necesitas una API key de Anthropic. "
                    "Obtén una en https://console.anthropic.com y configúrala así:\n\n"
                    "```bash\nexport ANTHROPIC_API_KEY='tu-api-key-aquí'\n```\n\n"
                    "Luego vuelve a correr `streamlit run cacao_montecarlo.py`"
                )
            except Exception as e:
                st.error(f"❌ Error al llamar a la IA: {e}")

    # ─────────────────────────────────────────
    # NOTAS METODOLÓGICAS
    # ─────────────────────────────────────────
    with st.expander("📖 Notas metodológicas"):
        st.markdown(f"""
        <div class='info-box'>
        <b>Modelo:</b> Geometric Brownian Motion · Shocks: {modelo_str}<br>
        <b>Semilla:</b> {semilla_str} · <b>Simulaciones:</b> {num_simulaciones}<br>
        <b>Drift diario calibrado:</b> {drift*100:.4f}% · <b>Volatilidad usada:</b> {volatilidad_ajustada*100:.2f}%<br>
        <b>VaR 95%:</b> pérdida máxima esperada con 95% de confianza al final del período.<br>
        <b>CVaR 95%:</b> pérdida promedio en el 5% de los peores escenarios (Expected Shortfall).<br>
        <b>Datos:</b> Futuros cacao ICE Nueva York (CC=F) · Último año disponible via yfinance.<br>
        <b>Importante:</b> Esta herramienta es de apoyo a la decisión, no constituye asesoría financiera.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Fórmula del modelo")
        st.latex(r"""
            P_{Ecu}(t) = \Biggl[ F(0) \cdot
            e^{\left(\mu - \frac{\sigma^2}{2}\right)t \;+\; \sigma W_t}
            - D \Biggr] \times K
        """)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class='info-box'>
            <b>F(0)</b> = Precio ICE hoy = <b>${precio_hoy:,.2f} USD/MT</b><br>
            <b>μ</b> = Drift diario = <b>{drift*100:.4f}%</b><br>
            <b>σ</b> = Volatilidad diaria = <b>{volatilidad_ajustada*100:.2f}%</b><br>
            <b>W<sub>t</sub></b> = Movimiento Browniano ({modelo_str})
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class='info-box'>
            <b>D</b> = Descuento logístico = <b>${descuento_tonelada:,} USD/MT</b><br>
            <b>K</b> = Factor de conversión = <b>{"0.04536 (qq/MT)" if "Quintal" in unidad else "1 (MT/MT)"}</b><br>
            <b>t</b> = Horizonte = <b>{dias_proyeccion} días</b><br>
            <b>N</b> = Simulaciones = <b>{num_simulaciones}</b>
            </div>
            """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"❌ Error al obtener datos o calcular simulaciones: {e}")
    st.info("Verifica tu conexión a internet o intenta más tarde. Los datos de futuros requieren conexión activa.")