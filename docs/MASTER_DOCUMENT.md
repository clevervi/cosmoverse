# COSMOVERSE — Documento Maestro del Proyecto

> **Instrucción para cualquier IA que lea este documento:** este es el estado consolidado y definitivo del proyecto COSMOVERSE tras dos rondas de revisión. Contiene la idea original, las correcciones aceptadas, las correcciones añadidas, la hipótesis reformulada, los resultados demostrables, el roadmap por etapas, el MVP publicable, el stack técnico, los riesgos con planes B y los criterios de éxito. Todo lo que aparece aquí es la versión vigente. Cualquier versión anterior queda superada.

---

## 1. Resumen ejecutivo

**COSMOVERSE** es un framework híbrido de inferencia cosmológica a gran escala que conecta emulación neuronal, marginalización explícita de incertidumbres bariónicas y observacionales, e inferencia bayesiana basada en simulación (SBI), con un módulo final de simulación diferenciable para inferencia de campo completo.

**No es** un simulador nuevo, ni un emulador nuevo, ni una plataforma lista para procesar datos reales de LSST/Euclid/DESI en su primera iteración.

**Sí es** un protocolo reproducible para conectar herramientas existentes (emuladores, módulos bariónicos, simuladores diferenciables, muestreadores) y demostrar que las posteriores resultantes están estadísticamente calibradas, con generalización cuantificada entre familias de simulación.

**Contribución científica central:** cuantificar la frontera sesgo–varianza al marginalizar sobre incertidumbres bariónicas y observacionales, y medir la degradación de la calibración al transferir entre familias de simulación. Esto es lo que distingue a COSMOVERSE de la literatura existente.

**MVP publicable (paper 1):** etapas 1–4 del roadmap con módulo observacional mínimo y calibración demostrada en datos sintéticos tipo Stage-IV.
**Extensión (paper 2):** etapa 5, inferencia de campo completo con JaxPM + MCLMC sobre datos sintéticos realistas.

---

## 2. Contexto y motivación

El cuello de botella que justifica el proyecto es real y medible:

- Una sola simulación de N-cuerpos de alta resolución consume de \(10^5\) a \(10^6\) horas de CPU.
- La inferencia bayesiana vía MCMC requiere millones de evaluaciones de verosimilitud.
- Los surveys de Stage-IV (LSST, Euclid, DESI) necesitan precisiones submétricas en escalas \(\lesssim 10\,\mathrm{Mpc}\), donde la física bariónica introduce sesgos si no se modela.
- La inferencia de campo completo (recuperar \(\delta_{\mathrm{ini}}(\mathbf{x})\) y parámetros cosmológicos simultáneamente) preserva información espacial que se pierde al comprimir en estadísticas resumidas, pero es computacionalmente exigente.

Un emulador que reduzca la evaluación del *forward model* a milisegundos no es un lujo: es condición necesaria para explotar los datos de Stage-IV. Pero un emulador solo no basta: se necesita propagar incertidumbres, modelar el proceso observacional y validar la calibración estadística.

---

## 3. Hipótesis científica (versión cuantitativa y vigente)

> **¿En qué régimen de escala y con qué presupuesto de simulación un framework híbrido —emulación + marginalización bariónica/observacional + simulación diferenciable— produce posteriores calibradas con sesgo residual por debajo de la precisión estadística de Stage-IV, y cuánto degrada la generalización al transferir entre familias de simulación?**

Esta formulación es medible y define criterios de éxito explícitos. Sustituye a la formulación original de sí/no ("¿es posible desarrollar…?"), que no era publicable porque su respuesta natural sería "sí, parcialmente".

**Nota importante:** ya existen trabajos que cubren partes de esta combinación (el benchmark de JaxPM incorpora distorsiones de espacio de redshifts, sesgo galáctico y ruido observacional en su modelo idealizado). Por tanto, **ningún módulo se presenta como completamente nuevo**. La novedad está en la integración, la propagación de incertidumbres y la demostración de calibración.

---

## 4. Resultados demostrables (los cuatro, en versión afinada)

### Resultado 1 — Precisión del emulador
El emulador conserva la precisión en estadísticas gaussianas **y no gaussianas**:
- MAPE en \(P(k)\) por debajo de umbrales explícitos para \(k < 1\,h\,\mathrm{Mpc}^{-1}\).
- Error acotado en *peak counts* y funcionales de Minkowski 3D.
- Validación en espacio real, espacio de redshifts, lentes débiles y funciones de masa de halos si el emulador las predice.

### Resultado 2 — Frontera sesgo–varianza (corrección clave)
No basta con decir "la marginalización reduce sesgos" (tautológico si el módulo funciona). El resultado científico es **cuantificar la frontera**:
> Cuánto sesgo se elimina y cuánto se infla el error al marginalizar sobre parámetros bariónicos y observacionales, y si esa frontera es compatible con la precisión de Stage-IV.

Se reporta la **curva sesgo–varianza completa**, no solo la reducción de sesgo.

### Resultado 3 — Calibración de la SBI
La SBI produce posteriores **calibradas** en datos sintéticos realistas:
- Tasas de cobertura explícitas vía **SBC** (Simulation-Based Calibration) y **TARP** (Tests of Accuracy with Random Points).
- Cobertura nominal del 68% / 95% verificada empíricamente.
- Detección de *misspecification* del modelo.

### Resultado 4 — Transferencia entre familias de simulación (contribución central)
Entrenar en una familia de simulación y evaluar en otra, cuantificando la degradación:
- Métricas de cobertura y sesgo antes y después de la transferencia.
- Acotado a una transferencia concreta (p. ej. N-body → hidrodinámico, o suite A → suite B), no "generalización entre familias" en abstracto.
- **Este es el resultado que distingue a COSMOVERSE de la literatura existente.** Se trata como contribución central, no como validación accesoria.
- **Presupuesto de calibración (v3.5):** en SIMBA, 25 simulaciones de calibración fallan la compuerta estricta de adaptación (cobertura 49% / 77%), mientras que 50 la satisfacen (58% / 91%, sesgo mediano 0.72σ). Por tanto, \(N_{\rm cal,min}=50\). El flujo fuente se entrena una sola vez y cada presupuesto parte de una copia independiente.

---

## 5. Arquitectura del sistema (cadena completa)

```
Simulación
    ↓
Emulación neuronal
    ↓
Física bariónica + astrofísica (feedback, galaxy bias, neutrinos)
    ↓
Modelo observacional (máscaras, selección, RSD, ruido, sistemáticas)
    ↓
    ├── SBI (Normalizing Flows) — exploración y posteriores
    └── Field-level (JaxPM + MCLMC) — inferencia fina
    ↓
Validación científica (SBC, TARP, cobertura, robustez, misspecification)
```

**Separación de tres niveles que no deben confundirse:**
1. La física del universo (simulación/emulación).
2. El proceso mediante el cual observamos el universo (módulo observacional).
3. El procedimiento estadístico de inferencia (SBI / MCLMC).

Esta separación es metodológica, no solo de software: mezclar niveles es fuente clásica de errores propagados indistinguibles.

---

## 6. Componentes técnicos detallados

### 6.1 Emulador cosmológico (forward model rápido)

**Arquitectura elegida para v1: TUNeS (U-Net temporal).**
- Entrenado con **solo 8 simulaciones de N-cuerpos**.
- Reproduce estadísticas gaussianas y no gaussianas (correlaciones de dos puntos, distribuciones de un punto, *peak counts*, funcionales de Minkowski 3D) con error de pocos por ciento en \(P(k)\) a \(k \simeq 1\,h\,\mathrm{Mpc}^{-1}\).
- Inferencia extremo a extremo \(256^3\) partículas → \(256^3\) grilla en **~25 s en una GPU**.
- Escala a volúmenes mayores mediante *batching* de partículas y refinamiento por ventanas.
- Referencia verificada: preprint arXiv, marzo 2026.

**Plan B / alternativa: CosmoRGNet (FNO informado por física).**
- Embebe estructura del grupo de renormalización funcional (FRG).
- MAPE **0.62%** para \(k < 1\,h\,\mathrm{Mpc}^{-1}\) en \(z=0\), **0.18%** en \(z=3\).
- Inferencia **12 ms** en una GPU.
- Validado contra Quijote Latin Hypercube y AbacusSummit.
- Predice espectros en espacio real, multipolos en espacio de redshifts, lentes débiles y funciones de masa de halos.
- **Estado: pendiente de verificación con DOI y manuscrito original antes de citarse como pilar.** Si no verifica, TUNeS cubre la etapa Foundation sin coste para el proyecto.

**Opcional para halos/galaxias: Mangrove (GNN sobre árboles de fusión).**
- Emula masa estelar, masa de gas frío, metalicidad, SFR instantánea y promediada, masa de agujeros negros.
- Error RMS hasta **2× menor** que otros métodos en caja de \((75\,h^{-1}\mathrm{Mpc})^3\).
- **40 s** de ejecución, 4 órdenes de magnitud más rápido que el modelo semianalítico de referencia.
- Referencia verificada (arXiv).

**Decisión de v1:** empezar con **una sola arquitectura**. No mezclar FNO + U-Net + GNN desde el inicio. La complejidad se añade solo cuando el esqueleto esté validado, para poder atribuir fallos a componentes concretos.

### 6.2 Módulo bariónico (marginalización obligatoria)

**Por qué es obligatorio:** la materia oscura fría no describe todos los observables de un survey real. Formación estelar, enfriamiento/calentamiento de gas, feedback de supernovas, feedback de AGN y redistribución de materia modifican la distribución en escalas no lineales. Ignorarlo sesga los parámetros cosmológicos en lentes débiles y escalas \(\lesssim 10\,\mathrm{Mpc}\).

**Herramientas operativas existentes (no hay que inventarlas):**

- **BCEmu** (basado en "baryonification"): versiones de 1, 3 y 7 parámetros (BCEmu1, BCEmu3, BCEmu7). Usado en análisis conjuntos de *cosmic shear* DES Y3 con kSZ de ACT. Más interpretable para marginalización.
- **BACCOemu**: emuladores de espectro de potencia lineal y no lineal **con física bariónica**. Usado en análisis de *cosmic shear* de DES Y3 para explotar escalas menores que el análisis oficial, modelando conjuntamente cosmología y feedback. Más amplio y ya validado en DES Y3.

**Decisión de v1: elegir UNO.** Integrar ambos en v1 es complejidad innecesaria que impide atribuir errores. Recomendación: **BCEmu** por interpretabilidad de la marginalización; BACCOemu entra en v2 o como validación cruzada.

**Datos de entrenamiento:** CAMELS segunda generación — **1,192 simulaciones de IllustrisTNG** en cajas de \((50\,h^{-1}\mathrm{Mpc})^3\), variando **35 parámetros** cosmológicos, astrofísicos y numéricos. Cobertura amplia del *prior*.

### 6.3 Módulo observacional (desde el principio, no al final)

**Este es el punto más importante y la corrección estructural del proyecto.**

Un FNO entrenado con campos periódicos no se convierte automáticamente en un modelo observacional de DESI. La cadena debe incluir explícitamente:

1. Máscaras y geometría del survey.
2. Funciones de selección.
3. Sesgos de galaxias (*galaxy bias*).
4. Errores de redshift y distorsiones en espacio de redshifts (RSD).
5. Ruido de disparo (*shot noise*).
6. Incompletitud observacional.
7. Sistemáticas instrumentales.
8. Efectos de proyección y calibración de lente gravitacional.

**Si el módulo observacional se pospone a la etapa final, el pipeline puede superar todas las validaciones internas y fallar precisamente en el paso que conecta con datos reales.** Debe integrarse **antes** de la SBI a gran escala, no después.

### 6.4 Inferencia — Fase A: SBI con Normalizing Flows

- Aprende la verosimilitud (o directamente la posterior) y genera muestras de la distribución posterior dados los datos.
- Aplicaciones validadas: **SimBIG** (SBI con *normalizing flows* para clustering de galaxias con *marked power spectra*, 2025), SBI híbrido simulación + teoría de campo efectivo (Zhang, Modi & Philcox, 2025), flujos continuos y autoregresivos enmascarados para estimación de posterior y verosimilitud.
- En esta fase el emulador rápido (TUNeS) actúa como *forward model* para explorar ampliamente el espacio de parámetros.

### 6.5 Inferencia — Fase B: field-level con JaxPM + MCLMC

- **JaxPM**: simulador de N-cuerpos diferenciable en JAX, distribuible en multi-GPU y multi-nodo.
- **MCLMC** (Microcanonical Langevin Monte Carlo): en el benchmark de Simon-Onfroy, Lanusse y de Mattia (JCAP, diciembre 2025) sobre un simulador diferenciable basado en JaxPM, **supera a los muestreadores ajustados por más de un orden de magnitud** en problemas de \(\ge 10^6\) dimensiones, con **sesgo mínimo en la posterior marginal**.
- **Cautela necesaria:** el benchmark **no demuestra** que cualquier survey real de LSST/Euclid/DESI pueda analizarse con ese rendimiento. Requiere:
  - Precondicionamiento adecuado de variables latentes (explícitamente señalado en el benchmark).
  - Reproducción de resultados en el problema propio.
  - Verificación de calibración y sesgo numérico.
- **Formulación correcta:** MCLMC es un **candidato prometedor** para field-level, sujeto a validación. No es una solución mágica.

### 6.6 Validación científica (no negociable)

- **SBC** (Simulation-Based Calibration): verifica que el 68% de las posteriores contengan el valor verdadero el 68% de las veces, etc.
- **TARP** (Tests of Accuracy with Random Points): cobertura empírica.
- **Validación contra estadísticas no gaussianas:** no basta con \(P(k)\). *Peak counts*, funcionales de Minkowski, distribuciones de un punto.
- **Detección de misspecification:** *normalizing flows* dependientes de escala (Akhmetzhanova, Cuesta-Lázaro et al., 2025).
- **Nota:** la cobertura sola no basta; la literatura reciente advierte que también hay que verificar el comportamiento a valores de parámetro fijos.

---

## 7. Protocolo de datos y partición (crítico)

**No usar división aleatoria 80/20.** Las simulaciones que comparten condiciones iniciales o cosmologías cercanas producen métricas de test engañosamente buenas por correlación train/test, no por generalización.

**Protocolo obligatorio:**
1. Partición por **condiciones iniciales** y **cosmología**.
2. Cuando sea posible, partición por **familia de simulación**.
3. **Tests de transferencia entre simuladores:** entrenar con una familia y evaluar con otra (p. ej. N-body → hidrodinámico, o suite A → suite B).
4. Reportar métricas de cobertura y sesgo antes/después de la transferencia.

Este protocolo es la base del Resultado 4 y debe quedar escrito en la etapa 1.

---

## 8. Roadmap por etapas (versión final)

| Etapa | Contenido | Correcciones aplicadas |
|---|---|---|
| **1. Foundation** | Emulador (TUNeS; CosmoRGNet solo si verifica), condiciones iniciales, benchmarks reproducibles | Partición por ICs/cosmología explícita; una sola arquitectura |
| **2. Physical uncertainty** | **Un** emulador bariónico (BCEmu **o** BACCOemu) + propagación de incertidumbres | No integrar ambos en v1 |
| **3. Observational realism** | Máscaras, selección, galaxy bias lineal, RSD, shot noise, sistemáticas | **Antes** de la SBI, no después |
| **4. Bayesian inference** | SBI + Normalizing Flows + SBC/TARP | Cobertura como criterio de aceptación |
| **5. Field-level reconstruction** | JaxPM + MCLMC (precondicionado, reproducido) | Extensión avanzada, no núcleo de v1 |
| **6. Scientific validation** | Datos independientes y, progresivamente, reales | Intercalada, no solo al final |

---

## 9. MVP publicable

**Paper 1 (metodológico) = Etapas 1–4 con módulo observacional mínimo.**
Demuestra: precisión del emulador, frontera sesgo–varianza con marginalización bariónica+observacional, calibración SBC/TARP en datos sintéticos tipo Stage-IV, y transferencia entre familias con degradación cuantificada.

**Paper 2 (extensión) = Etapa 5.**
Field-level con JaxPM + MCLMC sobre datos sintéticos realistas, con calibración verificada.

**No vender las dos juntas en la primera iteración.** Construir un emulador validado contra estadísticas no gaussianas, integrar un módulo bariónico con marginalización, añadir un modelo observacional realista y calibrar la SBI con SBC/TARP **ya constituye una contribución doctoral completa**. Field-level con datos reales es un segundo proyecto o la extensión postdoctoral.

---

## 10. Stack técnico

| Componente | Herramienta |
|---|---|
| Emulador | JAX (FNO) o PyTorch + U-Net temporal (TUNeS) |
| GNN (opcional halos/galaxias) | PyTorch + PyTorch Geometric (Mangrove) |
| SBI | `sbi` (Normalizing Flows) |
| Simulador diferenciable | JaxPM (JAX, multi-GPU/multi-nodo) |
| Muestreador field-level | MCLMC (precondicionado) |
| Emulación bariónica | BCEmu **o** BACCOemu (v1: uno solo) |
| Datos de entrenamiento | CAMELS 2ª gen (1,192 sims, 35 params), Quijote, AbacusSummit |
| Visualización | Three.js/WebGPU + FastAPI (capa de difusión, **no prioridad científica**) |

**Nota sobre visualización:** útil para comunicación científica, pero no es el cuello de botella científico ni la prioridad a gran escala. La prioridad es la validación estadística y la cuantificación de incertidumbres.

---

## 11. Posicionamiento frente a la literatura (qué NO somos)

- **No** somos un simulador nuevo.
- **No** somos un emulador nuevo (usamos TUNeS, CosmoRGNet, Mangrove).
- **No** somos un muestreador nuevo (usamos MCLMC).
- **No** somos una plataforma lista para datos reales de LSST/Euclid/DESI en v1.
- **No** prometemos descubrir nueva física.
- **No** afirmamos unicidad antes de una revisión sistemática de la literatura.

**Lo que sí somos:**
> Un protocolo reproducible para conectar emuladores, marginalización bariónica/observacional y simulación diferenciable, y demostrar que las posteriores resultantes están calibradas, con generalización cuantificada entre familias de simulación.

---

## 12. Riesgos y planes B

| Riesgo | Plan B |
|---|---|
| CosmoRGNet no verificable (sin DOI/manuscrito) | TUNeS cubre la etapa Foundation sin coste. Verificar antes de citar; no citar como pilar sin evidencia. |
| MCLMC no reproduce el benchmark en nuestro problema | Reportar la degradación como resultado; usar HMC/NUTS como comparación. El benchmark no garantiza el rendimiento en cualquier aplicación. |
| BCEmu insuficiente para nuestro régimen | Cambiar a BACCOemu (ya validado en DES Y3). Decisión reversible en v2. |
| Transferencia entre simuladores no acotada | Acotar a una transferencia concreta (N-body → hidrodinámico) y reportar métricas explícitas. |
| Módulo observacional demasiado costoso en v1 | Implementar versión mínima (máscaras + selección + bias lineal + RSD + shot noise) y declarar las sistemáticas restantes como trabajo futuro. |
| SBI mal calibrada por emulador sesgado | SBC/TARP como criterio de aceptación bloqueante; no avanzar de etapa sin cobertura verificada. |

---

## 13. Criterios de éxito medibles

1. **Emulador:** MAPE en \(P(k)\) por debajo de umbral explícito (\(<1\%\) a \(k<1\,h\,\mathrm{Mpc}^{-1}\)) y error acotado en *peak counts* y funcionales de Minkowski.
2. **Bariónico:** curva sesgo–varianza reportada; frontera compatible con Stage-IV identificada.
3. **SBI:** tasas de cobertura SBC/TARP dentro del intervalo nominal ± tolerancia declarada.
4. **Transferencia:** degradación de cobertura y sesgo cuantificada y acotada entre familias.
5. **Reproducibilidad:** pipeline ejecutable de extremo a extremo con benchmarks públicos.

---

## 14. Decisiones pendientes (a resolver en la etapa correspondiente)

- **CosmoRGNet:** verificar DOI y manuscrito antes de citar. Si no verifica, retirar de la propuesta y usar TUNeS.
- **BCEmu vs BACCOemu:** decisión de v1 (recomendación: BCEmu por interpretabilidad; BACCOemu en v2).
- **Transferencia concreta del Resultado 4:** definir la pareja de familias (N-body → hidrodinámico es la candidata natural por disponibilidad en CAMELS).
- **Alcance del módulo observacional mínimo:** decidir qué sistemáticas entran en v1 y cuáles se declaran como trabajo futuro.
- **Definición operativa de "compatible con Stage-IV":** fijar el umbral numérico de sesgo residual aceptable antes de empezar la etapa 4.

---

## 15. Glosario rápido (para IA)

- **SBI:** Simulation-Based Inference. Inferencia bayesiana cuando la verosimilitud no tiene forma analítica y se aproxima con simulaciones.
- **Normalizing Flows:** familia de modelos generativos que aprenden mapeos invertibles entre distribuciones; usados en SBI para aprender la posterior.
- **FNO:** Fourier Neural Operator. Operador neuronal que aprende mapeos entre espacios de funciones; generaliza entre resoluciones.
- **GNN:** Graph Neural Network. Red sobre grafos; natural para catálogos de halos/galaxias.
- **JaxPM:** simulador de N-cuerpos diferenciable en JAX; permite calcular gradientes exactos respecto a parámetros e ICs.
- **MCLMC:** Microcanonical Langevin Monte Carlo; muestreador de alta dimensión, sin corrección de Metropolis-Hastings, requiere precondicionamiento.
- **SBC:** Simulation-Based Calibration; test de cobertura de posteriores.
- **TARP:** Tests of Accuracy with Random Points; test empírico de cobertura.
- **Baryonification:** modelo paramétrico que introduce correcciones bariónicas en simulaciones de materia oscura.
- **CAMELS:** suite de simulaciones (N-body + hidrodinámicas) para entrenar ML. 2ª gen: 1,192 sims IllustrisTNG, 35 params, cajas de \((50\,h^{-1}\mathrm{Mpc})^3\).
- **Stage-IV:** generación de surveys cosmológicos (LSST, Euclid, DESI) con precisión submétrica en escalas no lineales.

---

## 16. Conclusión

COSMOVERSE es un proyecto de investigación metodológica con una tesis doctoral como vehículo natural. La primera fase —emulador validado + módulo bariónico + módulo observacional + SBI calibrada— ya es una contribución científica seria y publicable. La segunda fase —field-level con JaxPM + MCLMC— es la extensión avanzada. Lo que el proyecto **no** hace es vender la hoja de ruta como si ya estuviera recorrida.

**Meta final, en una frase:**
> Que un investigador pueda usar COSMOVERSE para plantear una pregunta sobre el universo, ejecutar un análisis y obtener resultados cuya fiabilidad pueda comprobar.

Eso es más valioso que otro proyecto de inteligencia artificial con una visualización espectacular del cosmos.

---

*Fin del documento maestro. Versión vigente tras dos rondas de revisión. Cualquier decisión posterior debe actualizar este documento, no sustituirlo por opiniones dispersas.*
