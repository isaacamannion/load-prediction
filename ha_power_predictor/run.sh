#!/usr/bin/env bashio

bashio::log.info "Starting HA Power Predictor..."

# Read configuration from options
export POWER_ENTITY=$(bashio::config 'power_entity')
export TEMPERATURE_ENTITY=$(bashio::config 'temperature_entity')
export BIN_SIZE_MINUTES=$(bashio::config 'bin_size_minutes')
export N_POWER_LAGS=$(bashio::config 'n_power_lags')
export N_TEMP_LAGS=$(bashio::config 'n_temp_lags')
export TRAIN_PERCENTAGE=$(bashio::config 'train_percentage')
export QUANTILE=$(bashio::config 'quantile')
export USE_DYNAMIC_QUANTILE=$(bashio::config 'use_dynamic_quantile')
export PEAK_START=$(bashio::config 'peak_start')
export PEAK_END=$(bashio::config 'peak_end')
export PEAK_QUANTILE=$(bashio::config 'peak_quantile')
export OFFPEAK_QUANTILE=$(bashio::config 'offpeak_quantile')
export HISTORY_DAYS=$(bashio::config 'history_days')
export TIMEZONE=$(bashio::config 'timezone')

# Get Home Assistant configuration
export SUPERVISOR_TOKEN="${SUPERVISOR_TOKEN}"
export HA_URL="http://supervisor/core"

bashio::log.info "Power Entity: ${POWER_ENTITY}"
bashio::log.info "Temperature Entity: ${TEMPERATURE_ENTITY}"
bashio::log.info "History Days: ${HISTORY_DAYS}"

# Start the Flask application
cd /app
exec gunicorn --bind 0.0.0.0:8099 --workers 2 --timeout 300 app.main:app
