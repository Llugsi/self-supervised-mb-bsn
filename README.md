# Self-Supervised Multi-Branch Blind-Spot Networks for Embedded Field Restoration

Repositorio oficial para el modelo de redes Multi-Branch Blind-Spot Network (MB-BSN) orientado a la restauración espacial de campos hidrodinámicos en tiempo real.

---

## 📌 Resumen del Proyecto
El marco implementa operadores topológicos de enmascaramiento para evitar fugas de datos y está optimizado para ejecutarse eficientemente en CPUs de borde con limitaciones de recursos.

---

## 📊 Descarga de Datos y Referencia
El conjunto de datos de velocidad de mecánica de fluidos (2D Navier-Stokes) se puede descargar manualmente desde el siguiente enlace:
* **Dataset:** [Kaggle Repository](https://www.kaggle.com/datasets/wenwangou/n5000nse?resource=download)

---

## 📂 Inventario del Repositorio
* `main_launcher.py`: Interfaz de línea de comandos para controlar la ejecución.
* `models.py` y `dataset.py`: Topología del modelo y cargador de datos.
* `train_real.py` y `train_synthetic.py`: Canales de entrenamiento real y sintético.
* `run_ablation.py` y `compile_scientific_tables.py`: Pruebas de ablación y compilador LaTeX.

---

## 🚀 Guía de Ejecución

### 1. Instalación de Requisitos
Clona el repositorio e instala las dependencias:
```bash
git clone https://github.com/Llugsi/
cd self-supervised-mb-bsn
pip install -r requirements.txt
```

### 2. Ejecución del Sistema
Inicia el orquestador principal:
```bash
python main_launcher.py
```
Selecciona las opciones en la terminal para entrenar, ejecutar pruebas o compilar tablas.

---

## 📜 Licencia
Licencia **MIT**.
