# Fruit Tracking

Sistema de detección y seguimiento de frutas utilizando técnicas de computer vision y algoritmos de tracking basados en Stone Soup. El proyecto incluye herramientas para procesar imágenes secuenciales, generar trayectorias de objetos y visualizar resultados tanto con detección automática como con ground truth.

## Instalación

### 1. Instalar Docker

```bash
# Actualizar el sistema
sudo apt update

# Instalar dependencias
sudo apt install apt-transport-https ca-certificates curl gnupg lsb-release

# Añadir la clave GPG oficial de Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Añadir el repositorio de Docker
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Instalar Docker
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

### 2. Configurar permisos de Docker

```bash
# Añadir tu usuario al grupo docker
sudo usermod -aG docker $USER

# Aplicar los cambios de grupo
newgrp docker

# Verificar que Docker funciona sin sudo
docker run hello-world
```

### 3. Instalar Visual Studio Code

Descargar e instalar VS Code desde [https://code.visualstudio.com/](https://code.visualstudio.com/)

### 4. Instalar extensiones necesarias

En VS Code, instala las siguientes extensiones:
- **Dev Containers** (ms-vscode-remote.remote-containers)
- **Docker** (ms-azuretools.vscode-docker)

## Abrir el entorno de desarrollo

1. Abrir el proyecto en VS Code:
   ```bash
   code /ruta/hacia/fruit-tracking
   ```

2. Abrir el Dev Container:
   - Presiona `Ctrl+Shift+P`
   - Escribe "Dev Containers: Open Folder in Container"
   - Selecciona la opción y espera a que el contenedor se construya

3. Una vez que VS Code se reabra dentro del contenedor, deberías ver "[Dev Container]" en la barra de estado inferior.

## Uso del sistema de tracking

### Tutorial completo con detección y tracking
Para ejecutar el tutorial completo que incluye detección y tracking, utiliza el notebook `Video_Processing.ipynb`. Este notebook realiza tanto la detección de objetos como el seguimiento en un flujo completo.

### Tracking con Ground Truth (GT)
Este proyecto también permite trabajar con Ground Truth existente ejecutando scripts individuales en el siguiente orden:

1. **Visualizar Ground Truth**
   ```bash
   python visualize_gt.py
   ```
   Genera un video con las imágenes originales y las anotaciones GT superpuestas.

2. **Ejecutar tracking**
   ```bash
   python coco_tracking.py
   ```
   Procesa las anotaciones GT y genera las trayectorias de tracking utilizando filtros de Kalman.

3. **Visualizar resultados del tracking**
   ```bash
   python visualize_tracking.py
   ```
   Crea un video mostrando las trayectorias generadas por el algoritmo de tracking.

### Configuración de paths
Antes de ejecutar los scripts, verifica y edita los paths en cada archivo según tu configuración local, o mantén la estructura de carpetas por defecto.

### Personalización del tracking
El script `coco_tracking.py` puede ser personalizado según las necesidades específicas:
- Modelos de transición (velocidad constante, random walk)
- Parámetros de ruido del proceso y medición
- Algoritmos de asociación de datos
- Métricas de distancia para la asociación
- Configuración del filtro de Kalman