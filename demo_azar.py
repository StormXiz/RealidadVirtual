"""
Demostración del Grupo 4: el azar del computador es una fórmula.

Ejemplo de Loja (supuestos propios, no son datos reales): un puesto de café
lojano junto al Parque Central prepara cada día 90 tazas. La demanda diaria
se modela con una distribución de Poisson de media 80. Cada taza vendida deja
$1,00 de ingreso y cada taza preparada cuesta $0,45. Una réplica simula un
mes de 30 días y devuelve la utilidad diaria promedio.

Uso:
    python demo_azar.py                 # valores por defecto
    python demo_azar.py --lam 95        # cambiar la demanda media
    python demo_azar.py --stock 110     # cambiar las tazas preparadas
    python demo_azar.py --n 10          # cambiar el número de réplicas
Requiere Python 3.8 o superior y NumPy 1.17 o superior.
"""
import argparse
import random
import secrets

import numpy as np

PRECIO, COSTO, DIAS = 1.00, 0.45, 30


def replica(rng, lam, stock):
    """Simula un mes y devuelve la utilidad diaria promedio."""
    demanda = rng.poisson(lam, DIAS)
    vendidas = np.minimum(demanda, stock)
    return float(np.mean(PRECIO * vendidas - COSTO * stock))


def replicas_mal(n, lam, stock, semilla):
    """ERROR típico: la semilla se fija dentro de cada réplica."""
    salidas = []
    for _ in range(n):
        rng = np.random.default_rng(semilla)   # mismo estado inicial cada vez
        salidas.append(replica(rng, lam, stock))
    return np.array(salidas)


def replicas_bien(n, lam, stock, semilla):
    """Correcto: un flujo independiente por réplica con SeedSequence.spawn."""
    hijos = np.random.SeedSequence(semilla).spawn(n)
    return np.array([replica(np.random.default_rng(h), lam, stock)
                     for h in hijos])


def resumen(nombre, x):
    print(f"{nombre:<28} media = {x.mean():7.4f}   desv. est. = {x.std(ddof=1):7.4f}"
          if len(x) > 1 else f"{nombre}: {x}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=80)
    ap.add_argument("--stock", type=int, default=90)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--semilla", type=int, default=2024)
    a = ap.parse_args()

    print("1. Reproducibilidad: la misma semilla da la misma secuencia")
    for _ in range(2):
        r = random.Random(a.semilla)
        print("   random (Mersenne Twister):", [r.randint(1, 6) for _ in range(6)])
    for _ in range(2):
        g = np.random.default_rng(a.semilla)
        print("   default_rng (PCG64):      ", g.integers(1, 7, 6).tolist())
    print("   Generador de numpy:", type(np.random.default_rng().bit_generator).__name__)

    print("\n2. Forma antigua frente a default_rng")
    np.random.seed(a.semilla)
    print("   np.random.seed (estado global):", np.random.randint(1, 7, 6).tolist())
    print("   default_rng (estado propio):   ",
          np.random.default_rng(a.semilla).integers(1, 7, 6).tolist())
    print("   Con la misma semilla los dos generadores producen secuencias distintas.")

    print(f"\n3. Réplicas (lam={a.lam}, stock={a.stock}, n={a.n})")
    mal = replicas_mal(a.n, a.lam, a.stock, a.semilla)
    bien = replicas_bien(a.n, a.lam, a.stock, a.semilla)
    print("   MAL  (semilla dentro de cada réplica):", np.round(mal, 4).tolist())
    resumen("   MAL", mal)
    print("   BIEN (SeedSequence.spawn):            ", np.round(bien, 4).tolist())
    resumen("   BIEN", bien)

    print("\n4. Para contraseñas se usa secrets, no random ni numpy")
    print("   token de 8 bytes:", secrets.token_hex(8), "(cambia en cada ejecución)")


if __name__ == "__main__":
    main()
