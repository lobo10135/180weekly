import datetime
import io
import os
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

reportlab_available = True
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
except ImportError:
    reportlab_available = False

# Streamlit Layout konfigurieren
st.set_page_config(page_title="180's Weekly Scanner", page_icon="🔮", layout="wide")


@st.cache_data(ttl=86400)
def get_sp500_tickers():
    """Lädt die aktuellen S&P 500 Ticker dynamisch von Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,"
            " like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    tables = pd.read_html(io.StringIO(response.text))
    df = tables[0]
    tickers = df["Symbol"].tolist()
    tickers = [t.replace(".", "-") for t in tickers]
    return tickers


@st.cache_data(ttl=86400)
def get_eurostoxx50_tickers():
    """Gibt eine bereinigte Liste aktueller Euro Stoxx 50 Unternehmen mit korrekten Yahoo-Finance-Endungen zurück."""
    return [
        "ADS.DE", "AD.AS", "AI.PA", "AIR.PA", "ALV.DE", "AMS.MC", "ASML.AS",
        "CS.PA", "BBVA.MC", "SAN.MC", "BAS.DE", "BAYN.DE", "BMW.DE", "BNP.PA",
        "CRG.IR", "BN.PA", "DTG.DE", "DB1.DE", "DTE.DE", "ENEL.MI", "ENI.MI",
        "EL.PA", "FLTR.IE", "IBE.MC", "ITX.MC", "IFX.DE", "INGA.AS", "ISP.MI",
        "OR.PA", "MC.PA", "MBG.DE", "MUV2.DE", "PRX.AS",
        "RI.PA", "SAF.PA", "SAN.PA", "SAP.DE", "SU.PA", "SIE.DE", "STLAM.MI",
        "TTE.PA", "DG.PA", "VOW3.DE", "UCG.MI", "UNA.AS", "VIE.PA"
    ]


@st.cache_data(ttl=86400)
def get_dax_tickers():
    """Gibt eine bereinigte, feste Liste aktueller DAX-Unternehmen mit korrekten Yahoo-Finance-Endungen zurück."""
    return [
        "ADS.DE", "AIR.PA", "ALV.DE", "BAS.DE", "BAYN.DE", "BEI.DE", "BMW.DE",
        "BNR.DE", "CBK.DE", "CON.DE", "DTG.DE", "DBK.DE", "DB1.DE", "DHL.DE",
        "DTE.DE", "EOAN.DE", "FRE.DE", "HNR1.DE", "HEI.DE", "HEN.DE", "IFX.DE",
        "MBG.DE", "MRK.DE", "MTX.DE", "MUV2.DE", "PAH3.DE", "QGEN.DE",
        "RHM.DE", "RWE.DE", "SAP.DE", "SIE.DE", "SHL.DE", "SY1.DE",
        "VOW3.DE", "VNA.DE", "ZAL.DE"
    ]


@st.cache_data(ttl=86400)
def get_mdax_tickers():
    """Gibt eine bereinigte, feste Liste aktueller MDAX-Unternehmen mit korrekten Yahoo-Finance-Endungen zurück."""
    return [
        "AIXA.DE", "ARL.DE", "AT1.DE", "BC8.DE", "BEI2.DE", "HOT.DE", "DBAN.DE",
        "COP.DE", "DBG.DE", "DHER.DE", "DWNI.DE", "EVT.DE", "FRA.DE", "FNT.DE",
        "G24.DE", "GKS.DE", "HAG.DE", "HNR.DE", "KKR.DE", "KRN.DE",
        "LAN.DE", "LEG.DE", "LHA.DE", "NEM.DE", "NDX1.DE", "PSM.DE", "RAT.DE",
        "RXC.DE", "SANT.DE", "SHA.DE", "TEG.DE", "UN01.DE", "WAC.DE", "WAF.DE",
        "JEN.DE", "ETG.DE", "NXR.DE", "PSM.DE", "BYW.DE", "GBF.DE", "SZG.DE"
    ]


def evaluate_stock_weekly(df_single, ticker, strategy_mode):
    """Prüft eine einzelne Wochenchart-Historie auf Long- oder Short-Strategie."""
    try:
        if df_single is None or len(df_single) < 55:
            return None

        if isinstance(df_single.columns, pd.MultiIndex):
            df_single.columns = df_single.columns.get_level_values(0)

        df_weekly = df_single.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }).dropna()

        if len(df_weekly) < 55:
            return None

        df_weekly["SMA10"] = df_weekly["Close"].rolling(window=10).mean()
        df_weekly["SMA50"] = df_weekly["Close"].rolling(window=50).mean()

        w = df_weekly.iloc[-1]
        w_minus_1 = df_weekly.iloc[-2]

        range_w_1 = w_minus_1["High"] - w_minus_1["Low"]
        if range_w_1 == 0 or pd.isna(range_w_1):
            return None
        
        range_w = w["High"] - w["Low"]
        if range_w == 0 or pd.isna(range_w):
            return None

        if strategy_mode == "Long":
            lower_quartile_w_1 = w_minus_1["Low"] + 0.25 * range_w_1
            cond_w_minus_1 = w_minus_1["Close"] <= lower_quartile_w_1

            upper_quartile_w = w["Low"] + 0.75 * range_w
            cond_w_upper = w["Close"] >= upper_quartile_w
            cond_sma10 = w["Close"] > w["SMA10"]
            cond_sma50 = w["Close"] > w["SMA50"]

            if cond_w_minus_1 and cond_w_upper and cond_sma10 and cond_sma50:
                high_w = float(w["High"])
                stop_buy = round(high_w + 0.125, 2)
                stop_loss = round(stop_buy - 1.0, 2)

                return {
                    "Ticker": ticker,
                    "Stop Buy": stop_buy,
                    "Stop": stop_loss,
                    "Schluss (W)": round(float(w["Close"]), 2),
                    "High (W)": round(high_w, 2),
                    "Low (W)": round(float(w["Low"]), 2),
                    "SMA 10": round(float(w["SMA10"]), 2),
                    "SMA 50": round(float(w["SMA50"]), 2),
                    "Schluss (W-1)": round(float(w_minus_1["Close"]), 2),
                }
        elif strategy_mode == "Short":
            upper_quartile_w_1 = w_minus_1["Low"] + 0.75 * range_w_1
            cond_w_minus_1 = w_minus_1["Close"] >= upper_quartile_w_1

            lower_quartile_w = w["Low"] + 0.25 * range_w
            cond_w_lower = w["Close"] <= lower_quartile_w
            cond_sma10 = w["Close"] < w["SMA10"]
            cond_sma50 = w["Close"] < w["SMA50"]

            if cond_w_minus_1 and cond_w_lower and cond_sma10 and cond_sma50:
                low_w = float(w["Low"])
                stop_buy = round(low_w - 0.125, 2)
                stop_loss = round(stop_buy + 1.0, 2)

                return {
                    "Ticker": ticker,
                    "Stop Buy": stop_buy,
                    "Stop": stop_loss,
                    "Schluss (W)": round(float(w["Close"]), 2),
                    "High (W)": round(float(w["High"]), 2),
                    "Low (W)": round(low_w, 2),
                    "SMA 10": round(float(w["SMA10"]), 2),
                    "SMA 50": round(float(w["SMA50"]), 2),
                    "Schluss (W-1)": round(float(w_minus_1["Close"]), 2),
                }
    except Exception:
        return None
    return None


def generate_pdf(df, strategy_title):
    """Erstellt ein sauberes PDF im Querformat aus dem Ergebnis-DataFrame"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30,
    )
    
    elements = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=15,
    )
    
    subtitle_style = ParagraphStyle(
        'ReportSubtitle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.gray,
        spaceAfter=20,
    )

    elements.append(Paragraph(f"180's Weekly Scanner – Ergebnis ({strategy_title})", title_style))
    elements.append(Paragraph(f"Erstellt am: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
    
    table_data = [list(df.columns)]
    for _, row in df.iterrows():
        table_data.append([str(val) for val in row])
        
    t = Table(table_data, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f9f9f9')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d3d3d3')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
    ]))
    
    elements.append(t)
    doc.build(elements)
    buffer.seek(0)
    return buffer


# --- Benutzeroberfläche mit zentriertem Layout ---
_, col_center, _ = st.columns([1, 2, 1])

with col_center:
    if os.path.exists("bulle.jpg"):
        st.image("bulle.jpg", use_container_width=True)

    st.markdown("### 🔮 180's Weekly Scanner")
    st.markdown("""
    Dieser Streamlit-Scanner überprüft Aktien auf Basis von **Wochencharts** auf die **"180's"-Strategien**:

    * **🟢 Long-Strategie (Wochenbasis):**
      * **Woche W-1 (Vorwoche):** Schlusskurs im **unteren Viertel (untere 25 %)**.
      * **Woche W (Aktuelle Woche):** Schlusskurs im **oberen Viertel (obere 25 %)** **UND** über **10-Wochen** sowie **50-Wochen-Durchschnitt**.
      * **Berechnungen:** Stop Buy = High (Woche W) $+ 0.125$ | Stop = Stop Buy $- 1$

    * **🔴 Short-Strategie (Wochenbasis):**
      * **Woche W-1 (Vorwoche):** Schlusskurs im **oberen Viertel (obere 25 %)**.
      * **Woche W (Aktuelle Woche):** Schlusskurs im **unteren Viertel (untere 25 %)** **UND** unter **10-Wochen** sowie **50-Wochen-Durchschnitt**.
      * **Berechnungen:** Stop Buy = Low (Woche W) $- 0.125$ | Stop = Stop Buy $+ 1$
    """)
    st.write("---")

    st.subheader("S&P 500 Scanner")
    col1, col2 = st.columns(2)
    with col1:
        run_sp_long = st.button("🟢 S&P500 Long (Weekly)", type="secondary", use_container_width=True)
    with col2:
        run_sp_short = st.button("🔻 S&P500 Short (Weekly)", type="secondary", use_container_width=True)

    st.subheader("Euro Stoxx 50 Scanner")
    col_es1, col_es2 = st.columns(2)
    with col_es1:
        run_estoxx_long = st.button("🟢 Euro Stoxx 50 Long", type="secondary", use_container_width=True)
    with col_es2:
        run_estoxx_short = st.button("🔻 Euro Stoxx 50 Short", type="secondary", use_container_width=True)

    st.subheader("DAX Scanner")
    col3, col4 = st.columns(2)
    with col3:
        run_dax_long = st.button("🟢 Dax Long (Weekly)", type="secondary", use_container_width=True)
    with col4:
        run_dax_short = st.button("🔻 Dax Short (Weekly)", type="secondary", use_container_width=True)

    st.subheader("MDAX Scanner")
    col5, col6 = st.columns(2)
    with col5:
        run_mdax_long = st.button("🟢 MDax Long (Weekly)", type="secondary", use_container_width=True)
    with col6:
        run_mdax_short = st.button("🔻 MDax Short (Weekly)", type="secondary", use_container_width=True)

    triggered_button = None
    strategy_mode = None
    universe_type = None

    if run_sp_long:
        triggered_button, strategy_mode, universe_type = run_sp_long, "Long", "S&P 500"
    elif run_sp_short:
        triggered_button, strategy_mode, universe_type = run_sp_short, "Short", "S&P 500"
    elif run_estoxx_long:
        triggered_button, strategy_mode, universe_type = run_estoxx_long, "Long", "Euro Stoxx 50"
    elif run_estoxx_short:
        triggered_button, strategy_mode, universe_type = run_estoxx_short, "Short", "Euro Stoxx 50"
    elif run_dax_long:
        triggered_button, strategy_mode, universe_type = run_dax_long, "Long", "DAX"
    elif run_dax_short:
        triggered_button, strategy_mode, universe_type = run_dax_short, "Short", "DAX"
    elif run_mdax_long:
        triggered_button, strategy_mode, universe_type = run_mdax_long, "Long", "MDAX"
    elif run_mdax_short:
        triggered_button, strategy_mode, universe_type = run_mdax_short, "Short", "MDAX"

    if triggered_button:
        if universe_type == "S&P 500":
            tickers = get_sp500_tickers()
        elif universe_type == "Euro Stoxx 50":
            tickers = get_eurostoxx50_tickers()
        elif universe_type == "DAX":
            tickers = get_dax_tickers()
        else:
            tickers = get_mdax_tickers()

        st.info(
            f"Lade und analysiere {len(tickers)} Aktien aus dem **{universe_type}** auf Wochenbasis für die"
            f" **{strategy_mode}**-Strategie. Bitte warten..."
        )

        results = []
        progress_bar = st.progress(0)
        total = len(tickers)

        for idx, ticker in enumerate(tickers):
            try:
                df_single = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
                if df_single is not None and not df_single.empty:
                    res = evaluate_stock_weekly(df_single, ticker, strategy_mode)
                    if res:
                        results.append(res)
            except Exception:
                pass

            progress_bar.progress(min((idx + 1) / total, 1.0))

        progress_bar.empty()

        if results:
            st.success(
                f"Analyse abgeschlossen! Es wurden **{len(results)}** Treffer für"
                f" **{universe_type} {strategy_mode} (Weekly)** gefunden."
            )
            result_df = pd.DataFrame(results)
            
            desired_columns = [
                "Ticker",
                "Stop Buy",
                "Stop",
                "Schluss (W)",
                "High (W)",
                "Low (W)",
                "SMA 10",
                "SMA 50",
                "Schluss (W-1)",
            ]
            result_df = result_df[[col for col in desired_columns if col in result_df.columns]]

            st.dataframe(result_df, use_container_width=True, hide_index=True)
            
            if reportlab_available:
                pdf_buffer = generate_pdf(result_df, f"{universe_type} {strategy_mode} Weekly")
                st.download_button(
                    label=f"📥 Ergebnisse als PDF herunterladen ({universe_type} {strategy_mode} Weekly)",
                    data=pdf_buffer,
                    file_name=f"180s_weekly_{universe_type.replace(' ', '_').lower()}_{strategy_mode.lower()}_ergebnisse.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.info("Hinweis: Installieren Sie `reportlab` (`pip install reportlab`), um die PDF-Export-Funktion zu aktivieren.")
        else:
            st.warning(
                f"Aktuell erfüllen keine Aktien im **{universe_type}** auf Wochenbasis die Kriterien dieser"
                f" **{strategy_mode}**-Strategie."
            )
