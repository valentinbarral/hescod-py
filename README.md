# HESCOD (Python version)

**H**rramienta para la **E**valuación de **S**istemas de **CO**municaciones **D**igitales.

Simulador de canal con interfaz PyQt para:
- generar bits aleatorios o cargar bits desde imagen,
- elegir modulación,
- seleccionar codificación de canal,
- simular distintos tipos de canal,
- visualizar curvas y resultados.

## 1) Instalar `uv` (Python package manager)

### Linux / macOS

Instalación recomendada:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Comprobar instalación:

```bash
uv --version
```

### Windows (PowerShell)

Instalación recomendada:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Comprobar instalación:

```powershell
uv --version
```

## 2) Instalar dependencias del proyecto

Desde la carpeta del proyecto (`hescod-py`):

```bash
uv sync
```

## 3) Ejecutar la aplicación

```bash
uv run run_pyqt.py
```

## Licencia

Este proyecto se distribuye bajo licencia **MIT**.

## Autoría y créditos

- Autor de esta versión: **Valentín Barral** (UDC, Universidade da Coruña).
- Esta aplicación está basada en una versión en MATLAB creada por **Óscar Fresnedo** (UDC, Universidade da Coruña).
