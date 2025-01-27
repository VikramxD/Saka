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
    if [ ! -f "${GRAFANA_DIR}/bin/grafana-server" ]; then
        wget "https://dl.grafana.com/oss/release/grafana-${GRAFANA_VERSION}.linux-amd64.tar.gz"
        tar xzf "grafana-${GRAFANA_VERSION}.linux-amd64.tar.gz"
        mv "grafana-${GRAFANA_VERSION}"/* "${GRAFANA_DIR}/"
        rm -rf "grafana-${GRAFANA_VERSION}" "grafana-${GRAFANA_VERSION}.linux-amd64.tar.gz"
    fi
    
    # Configure Grafana
    cat > "${GRAFANA_DIR}/conf/custom.ini" << EOL
[server]
http_port = 3000
protocol = http

[security]
admin_user = admin
admin_password = admin

[paths]
data = ${MONITORING_DIR}/data/grafana
logs = ${MONITORING_DIR}/data/grafana/logs
plugins = ${MONITORING_DIR}/data/grafana/plugins
provisioning = ${GRAFANA_DIR}/conf/provisioning

[auth.anonymous]
enabled = true
org_role = Viewer
EOL

    # Copy Grafana configurations
    cp "${PROJECT_ROOT}/configs/grafana/provisioning/datasources/prometheus.yml" "${GRAFANA_DIR}/conf/provisioning/datasources/"
    cp "${PROJECT_ROOT}/configs/grafana/provisioning/dashboards/video-enhancer.yml" "${GRAFANA_DIR}/conf/provisioning/dashboards/"
    cp "${PROJECT_ROOT}/configs/grafana/dashboards/video_enhancer.json" "${GRAFANA_DIR}/dashboards/"
}

# Function to create start script
create_start_script() {
    echo "Creating start script..."
    cat > "${MONITORING_DIR}/start_monitoring.sh" << 'EOL'
#!/bin/bash
set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
PROMETHEUS_DIR="${SCRIPT_DIR}/prometheus"
GRAFANA_DIR="${SCRIPT_DIR}/grafana"

# Function to cleanup on exit
cleanup() {
    echo "Stopping monitoring stack..."
    kill $PROMETHEUS_PID $GRAFANA_PID 2>/dev/null || true
}

# Register cleanup function
trap cleanup EXIT INT TERM

# Cleanup any existing processes
pkill prometheus || true
pkill grafana-server || true
sleep 2

# Start Prometheus
echo "Starting Prometheus..."
cd "${PROMETHEUS_DIR}"
"${PROMETHEUS_DIR}/prometheus" \
    --config.file="${PROMETHEUS_DIR}/prometheus.yml" \
    --storage.tsdb.path="${SCRIPT_DIR}/data/prometheus" \
    --web.listen-address="0.0.0.0:9090" \
    --web.enable-lifecycle \
    --web.enable-admin-api &
PROMETHEUS_PID=$!

# Wait for Prometheus to start
echo "Waiting for Prometheus to start..."
for i in {1..30}; do
    if curl -s http://localhost:9090/-/healthy > /dev/null; then
        echo "Prometheus is ready"
        break
    fi
    sleep 1
done

# Start Grafana
echo "Starting Grafana..."
cd "${GRAFANA_DIR}"
"${GRAFANA_DIR}/bin/grafana-server" \
    --config="${GRAFANA_DIR}/conf/custom.ini" \
    --homepath="${GRAFANA_DIR}" &
GRAFANA_PID=$!

# Wait for Grafana to start
echo "Waiting for Grafana to start..."
for i in {1..30}; do
    if curl -s http://localhost:3000/api/health > /dev/null; then
        echo "Grafana is ready"
        break
    fi
    sleep 1
done

echo "Monitoring stack started!"
echo "Prometheus: http://localhost:9090"
echo "Grafana: http://localhost:3000 (login: admin/admin)"
echo "Press Ctrl+C to stop..."

# Wait for processes
wait $PROMETHEUS_PID $GRAFANA_PID
EOL

    chmod +x "${MONITORING_DIR}/start_monitoring.sh"
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
