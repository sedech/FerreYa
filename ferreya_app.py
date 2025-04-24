import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Configuración de la app
st.set_page_config(page_title="FerreYa", layout="wide")
st.title("🔩 FerreYa - Asistente Inteligente de Stock para Ferreterías")
st.markdown("Subí tu archivo de **stock actual** y tu archivo de **ventas**, y FerreYa te dirá qué reponer, qué falta y qué se vende más.")

# Subida de archivos
stock_file = st.file_uploader("📦 Cargar archivo de STOCK", type=["csv", "xlsx"])
ventas_file = st.file_uploader("🧾 Cargar archivo de VENTAS", type=["csv", "xlsx"])

if stock_file is not None and ventas_file is not None:
    # Leer archivos
    df_stock = pd.read_csv(stock_file)
    df_ventas = pd.read_csv(ventas_file)

    st.subheader("📋 Datos cargados")
    st.write("🔧 Stock:")
    st.dataframe(df_stock)
    st.write("🛒 Ventas:")
    st.dataframe(df_ventas)

    # Agrupar ventas
    ventas_agrupadas = df_ventas.groupby("producto")["cantidad"].sum().reset_index()
    ventas_agrupadas.rename(columns={"cantidad": "vendido_total"}, inplace=True)

    # Fusionar stock con ventas
    df_merged = pd.merge(df_stock, ventas_agrupadas, on="producto", how="left")
    df_merged["vendido_total"].fillna(0, inplace=True)

    # Calcular sugerencia de stock
    df_merged["stock_sugerido"] = df_merged["vendido_total"] * 1.5
    df_merged["a_reponer"] = df_merged["stock_sugerido"] - df_merged["stock_actual"]
    df_merged["a_reponer"] = df_merged["a_reponer"].apply(lambda x: max(x, 0))

    # Mostrar productos a reponer
    st.subheader("🚨 Productos a Reponer")
    productos_reponer = df_merged[df_merged["a_reponer"] > 0][["producto", "stock_actual", "vendido_total", "a_reponer"]]
    if not productos_reponer.empty:
        st.dataframe(productos_reponer)
        st.download_button("⬇️ Descargar lista de reposición", productos_reponer.to_csv(index=False).encode(), "reponer.csv", "text/csv")
    else:
        st.success("✅ Todo el stock está bien abastecido.")

    # Ranking de más vendidos
    st.subheader("🏆 Ranking de Productos Más Vendidos")
    st.dataframe(ventas_agrupadas.sort_values("vendido_total", ascending=False))

    # Visualización
    st.subheader("📊 Gráfico de Ventas")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x="producto", y="vendido_total", data=ventas_agrupadas, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    st.pyplot(fig)

    # Predicción simple de ventas
    st.subheader("📈 Predicción de Ventas Futuras")
    ventas_agrupadas["codigo_producto"] = ventas_agrupadas["producto"].astype("category").cat.codes
    X = ventas_agrupadas[["codigo_producto"]]
    y = ventas_agrupadas["vendido_total"]
    model = LinearRegression().fit(X, y)
    ventas_agrupadas["prediccion"] = model.predict(X)
    st.dataframe(ventas_agrupadas[["producto", "vendido_total", "prediccion"]])

    # Márgenes (estimado usando un margen del 30%)
    st.subheader("💰 Reporte de Márgenes Estimados")
    df_merged["precio_costo_estimado"] = df_merged["precio"] * 0.7  # Suponemos margen de 30%
    df_merged["margen_estimado"] = df_merged["precio"] - df_merged["precio_costo_estimado"]
    st.dataframe(df_merged[["producto", "precio", "precio_costo_estimado", "margen_estimado"]])
    st.download_button("⬇️ Descargar márgenes estimados", df_merged[["producto", "precio", "precio_costo_estimado", "margen_estimado"]].to_csv(index=False).encode(), "margenes.csv", "text/csv")
