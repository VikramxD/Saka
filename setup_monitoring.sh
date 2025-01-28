#!/bin/bash
set -e

# Configuration
PROMETHEUS_VERSION="2.45.0"
GRAFANA_VERSION="10.0.3"
PROJECT_ROOT="$(pwd)"
MONITORING_DIR="${PROJECT_ROOT}/monitoring"
PROMETHEUS_DIR="${MONITORING_DIR}/prometheus"
GRAFANA_DIR="${MONITORING_DIR}/grafana"

# Function to check if a port is in use
check_port() {
    local port=$1
    if netstat -tln | grep -q ":${port} "; then
        return 0
    else
        return 1
    fi
}

# Function to cleanup existing processes
cleanup_processes() {
    echo "Cleaning up existing processes..."
    
    # Kill any existing Prometheus process
    if pgrep prometheus > /dev/null; then
        echo "Stopping existing Prometheus process..."
        pkill prometheus || true
    fi
    
    # Kill any existing Grafana process
    if pgrep grafana-server > /dev/null; then
        echo "Stopping existing Grafana process..."
        pkill grafana-server || true
    fi
    
    # Wait for processes to stop
    sleep 2
    
    # Clean up data directories
    echo "Cleaning up data directories..."
    rm -rf "${MONITORING_DIR}/data/prometheus/*"
    rm -rf "${MONITORING_DIR}/data/grafana/*"
}

# Function to create necessary directories
create_directories() {
    echo "Creating directories..."
    mkdir -p "${PROMETHEUS_DIR}" "${GRAFANA_DIR}" \
        "${MONITORING_DIR}/data/prometheus" "${MONITORING_DIR}/data/grafana" \
        "${GRAFANA_DIR}/conf/provisioning/datasources" \
        "${GRAFANA_DIR}/conf/provisioning/dashboards" \
        "${GRAFANA_DIR}/dashboards"
}

# Function to setup Prometheus
setup_prometheus() {
    echo "Setting up Prometheus..."
    if [ ! -f "${PROMETHEUS_DIR}/prometheus" ]; then
        wget "https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.linux-amd64.tar.gz"
        tar xzf "prometheus-${PROMETHEUS_VERSION}.linux-amd64.tar.gz"
        mv "prometheus-${PROMETHEUS_VERSION}.linux-amd64"/* "${PROMETHEUS_DIR}/"
        rm -rf "prometheus-${PROMETHEUS_VERSION}.linux-amd64" "prometheus-${PROMETHEUS_VERSION}.linux-amd64.tar.gz"
    fi
    
    # Copy Prometheus config
    cp "${PROJECT_ROOT}/configs/prometheus/prometheus.yml" "${PROMETHEUS_DIR}/prometheus.yml"
}

# Function to setup Grafana
setup_grafana() {
    echo "Setting up Grafana..."
    
    # Download and extract Grafana
    wget "https://dl.grafana.com/oss/release/grafana-${GRAFANA_VERSION}.linux-amd64.tar.gz"
    tar -zxf "grafana-${GRAFANA_VERSION}.linux-amd64.tar.gz"
    
    # Create Grafana directories
    mkdir -p "${GRAFANA_DIR}/conf/provisioning/datasources"
    mkdir -p "${GRAFANA_DIR}/conf/provisioning/dashboards"
    mkdir -p "${GRAFANA_DIR}/dashboards"
    mkdir -p "${GRAFANA_DIR}/data"
    
    # Copy Grafana files
    cp -r "grafana-${GRAFANA_VERSION}"/* "${GRAFANA_DIR}/"
    
    # Copy existing configurations
    cp -r "${PROJECT_ROOT}/configs/grafana/provisioning/"* "${GRAFANA_DIR}/conf/provisioning/"
    cp -r "${PROJECT_ROOT}/configs/grafana/dashboards/"* "${GRAFANA_DIR}/dashboards/"
    
    # Create custom Grafana config
    cat > "${GRAFANA_DIR}/conf/custom.ini" << EOL
[paths]
data = ${GRAFANA_DIR}/data
logs = ${GRAFANA_DIR}/data/log
plugins = ${GRAFANA_DIR}/data/plugins
provisioning = ${GRAFANA_DIR}/conf/provisioning

[server]
protocol = http
http_port = 3000

[security]
admin_user = admin
admin_password = admin

[auth.anonymous]
enabled = true
org_role = Viewer
EOL

    # Cleanup
    rm -f "grafana-${GRAFANA_VERSION}.linux-amd64.tar.gz"
    rm -rf "grafana-${GRAFANA_VERSION}"
}

# Function to create start script
create_start_script() {
    echo "Creating start script..."
    cat > "${MONITORING_DIR}/start_monitoring.sh" << 'EOL'
#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

# Start Prometheus
./prometheus/prometheus \
    --config.file=prometheus/prometheus.yml \
    --storage.tsdb.path=data/prometheus \
    --web.console.libraries=prometheus/console_libraries \
    --web.console.templates=prometheus/consoles \
    --web.listen-address=:9090 &

# Start Grafana
./grafana/bin/grafana server \
    --homepath="${SCRIPT_DIR}/grafana" \
    --config="${SCRIPT_DIR}/grafana/conf/custom.ini" \
    &

echo "Monitoring stack started!"
echo "Prometheus: http://localhost:9090"
echo "Grafana: http://localhost:3000 (login: admin/admin)"
echo "Press Ctrl+C to stop..."

wait
EOL
}

# Main setup process
echo "Starting monitoring setup..."

# Run cleanup
cleanup_processes

# Create directories
create_directories

# Setup components
setup_prometheus
setup_grafana

# Create start script
create_start_script

echo "Setup complete! To start monitoring, run:"
echo "${MONITORING_DIR}/start_monitoring.sh"
