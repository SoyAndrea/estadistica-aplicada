# -*- coding: utf-8 -*-
"""Ganancia de negocio de una campaña de retención, en función del CORTE de probabilidad.

Idea: accionamos (ofrecemos una bonificación de retención) a los clientes con
probabilidad de baja >= corte. Cada decisión tiene un valor en pesos:

  matriz de ganancia por cliente
  ------------------------------------------------------------------
  accioná y ERA baja  (TP): + GANANCIA_RETENER - COSTO_ACCION   (lo retenemos)
  accioná y NO era    (FP): - COSTO_ACCION                      (gastamos al pedo)
  no accioná y ERA    (FN): - PERDIDA_NO_DETECTAR               (se va sin acción)
  no accioná y NO era (TN):   0

El corte óptimo teórico es  p* = COSTO_ACCION / (GANANCIA_RETENER + PERDIDA_NO_DETECTAR).

Uso:
  # con probabilidades y verdad (el profe con la solución; el alumno con SU hold-out):
  python ganancia.py --pred entregas/g3.csv --y kit/solucion_privada.csv --plot curva.png
  # sin argumentos: corre el EJEMPLO (xgboost baseline vs solución) y gudarda ejemplo_ganancia.png
"""
import argparse, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- parámetros del negocio (editables) ----
GANANCIA_RETENER   = 100000   # margen que se salva al retener a un churner accionado
COSTO_ACCION       = 80000    # bonificación del servicio: 40.000 $/mes x 2 meses
PERDIDA_NO_DETECTAR= 100000   # margen que se pierde si un churner se va sin acción

TEAL="#00897B"; ORANGE="#F2792B"; DARK="#0F2A2E"; GRAY="#5C6B6A"
PRED_NAMES=["clasebinaria","revenue","y_prob","prob","proba","prediction","pred","y","target","score"]

def _col_pred(df):
    df=df.copy(); df.columns=[c.strip().lower() for c in df.columns]
    pc=next((c for c in PRED_NAMES if c in df.columns and c!="id"),None)
    if pc is None:
        num=[c for c in df.columns if c!="id" and pd.api.types.is_numeric_dtype(df[c])]; pc=num[0]
    return df[["id",pc]].rename(columns={pc:"p"})

def _col_y(df):
    df=df.copy(); df.columns=[c.strip().lower() for c in df.columns]
    tcol=[c for c in df.columns if c not in ("id","usage")][0]
    keep=["id",tcol]+(["usage"] if "usage" in df.columns else [])
    return df[keep].rename(columns={tcol:"y"})

def ganancia_en_corte(y, p, corte):
    act = p >= corte
    tp = np.sum(act & (y==1)); fp = np.sum(act & (y==0)); fn = np.sum(~act & (y==1))
    return tp*(GANANCIA_RETENER-COSTO_ACCION) + fp*(-COSTO_ACCION) + fn*(-PERDIDA_NO_DETECTAR)

def curva(y, p, cortes=None):
    if cortes is None: cortes = np.linspace(0,1,101)
    g = np.array([ganancia_en_corte(y,p,c) for c in cortes])
    return cortes, g

def corte_optimo(y, p):
    cortes,g = curva(y,p); i=int(g.argmax()); return float(cortes[i]), float(g[i])

def graficar(y, p, path, titulo="Ganancia según el corte de probabilidad"):
    cortes,g = curva(y,p); i=int(g.argmax()); cstar=COSTO_ACCION/(GANANCIA_RETENER+PERDIDA_NO_DETECTAR)
    fig,ax=plt.subplots(figsize=(8.5,4.4))
    ax.plot(cortes, g/1e6, color=TEAL, lw=2.5)
    ax.axhline(0, color=GRAY, lw=1, ls=":")
    ax.scatter([cortes[i]],[g[i]/1e6], color=ORANGE, zorder=5, s=60,
               label=f"óptimo: corte {cortes[i]:.2f} → ${g[i]/1e6:,.1f} M")
    ax.axvline(cstar, color=GRAY, ls="--", lw=1.2, label=f"teórico p* = {cstar:.2f}")
    ymax=g[i]/1e6
    ax.set_ylim(ymax-150, ymax+18)   # enfocamos la zona útil (el extremo "accionar a todos" queda fuera de escala)
    ax.set_title(titulo, color=DARK, fontweight="bold")
    ax.set_xlabel("corte de probabilidad (accionamos si p ≥ corte)")
    ax.set_ylabel("ganancia de la campaña (millones $)")
    ax.annotate("accionar a todos (corte 0)\ndaría una pérdida enorme (fuera de escala)",
                xy=(0.02, (g[0]/1e6)), xytext=(0.12, ymax-130), fontsize=8.5, color=GRAY,
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1))
    ax.legend(frameon=False, loc="lower center"); ax.grid(alpha=.25)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pred"); ap.add_argument("--y", dest="ysol")
    ap.add_argument("--corte", type=float); ap.add_argument("--plot", default="curva_ganancia.png")
    a=ap.parse_args()
    if not a.pred:  # EJEMPLO
        a.pred, a.ysol, a.plot = "entregas/xgboost.csv", "kit/solucion_privada.csv", "ejemplo_ganancia.png"
        print("(ejemplo con el baseline XGBoost)")
    sub=_col_pred(pd.read_csv(a.pred)); sol=_col_y(pd.read_csv(a.ysol))
    m=sol.merge(sub,on="id"); y=m["y"].values; p=np.clip(m["p"].values,0,1)
    print(f"Parámetros: retener +${GANANCIA_RETENER:,} · costo acción -${COSTO_ACCION:,} · no detectar -${PERDIDA_NO_DETECTAR:,}")
    print(f"Corte teórico p* = {COSTO_ACCION/(GANANCIA_RETENER+PERDIDA_NO_DETECTAR):.2f}")
    c,gmax = corte_optimo(y,p)
    print(f"Corte óptimo (en estos datos): {c:.2f}  ->  ganancia máx = ${gmax:,.0f}")
    for cc in (0.5,):
        print(f"Ganancia con corte {cc}: ${ganancia_en_corte(y,p,cc):,.0f}")
    if a.corte is not None:
        print(f"Ganancia con TU corte {a.corte}: ${ganancia_en_corte(y,p,a.corte):,.0f}")
    graficar(y,p,a.plot); print("Gráfico:", a.plot)

if __name__=="__main__":
    main()
