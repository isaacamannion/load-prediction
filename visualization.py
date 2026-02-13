"""
Visualization functions for energy load prediction results.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_overview(df_test, y_test, y_pred_lr, y_pred_nn, lr_metrics, nn_metrics,
                  features, y_pred_lr_smooth=None, y_pred_nn_smooth=None,
                  smooth_metrics_lr=None, smooth_metrics_nn=None,
                  is_iterative=False, quantile=0.5, quantile_config=None):
    """
    Create overview plot showing full test period.
    Handles both single and dual model configurations.
    """
    timestamps = df_test['timestamp'].values

    # Determine how many subplots we need
    has_qr = y_pred_lr is not None
    has_nn = y_pred_nn is not None

    if has_qr and has_nn:
        # Both models - use 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        axes = [ax1, ax2]
    else:
        # Single model - use 1 subplot
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        axes = [ax]

    fig.suptitle('Energy Load Prediction - Full Test Period Overview', fontsize=16, fontweight='bold')

    # Plot Quantile Regression if available
    if has_qr:
        ax_idx = 0 if has_qr and has_nn else 0
        axes[ax_idx].plot(timestamps, y_test, 'k-', alpha=0.6, linewidth=1.5, label='Actual')
        if y_pred_lr_smooth is not None:
            axes[ax_idx].plot(timestamps, y_pred_lr, 'b--', alpha=0.3, linewidth=1, label='QR Raw')
            axes[ax_idx].plot(timestamps, y_pred_lr_smooth, 'b-', linewidth=2, label='QR Smoothed')
        else:
            axes[ax_idx].plot(timestamps, y_pred_lr, 'b-', linewidth=2, label='QR Prediction')

        if quantile_config is not None:
            peak_start = quantile_config['peak_start']
            peak_end = quantile_config['peak_end']
            peak_q = quantile_config['peak_quantile']
            offpeak_q = quantile_config['offpeak_quantile']
            title_q = f'Dynamic Q (Peak: {peak_q * 100:.0f}%, Off: {offpeak_q * 100:.0f}%)'
        else:
            title_q = f'{quantile * 100:.0f}th Percentile'

        coverage = lr_metrics.get('coverage', 0)
        metrics_text = f"R²={lr_metrics['r2']:.3f}, RMSE={lr_metrics['rmse']:.2f}kW, Cov={coverage}"
        axes[ax_idx].set_title(f'Quantile Regression ({title_q}) | {metrics_text}', fontsize=12, fontweight='bold')
        axes[ax_idx].set_ylabel('Power (kW)', fontsize=11)
        axes[ax_idx].legend(loc='upper left', fontsize=10)
        axes[ax_idx].grid(True, alpha=0.3)
        axes[ax_idx].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axes[ax_idx].xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(timestamps) // 2880 // 10)))
        if not has_nn:
            axes[ax_idx].set_xlabel('Date', fontsize=11)

    # Plot Neural Network if available
    if has_nn:
        ax_idx = 1 if has_qr and has_nn else 0
        axes[ax_idx].plot(timestamps, y_test, 'k-', alpha=0.6, linewidth=1.5, label='Actual')
        if y_pred_nn_smooth is not None:
            axes[ax_idx].plot(timestamps, y_pred_nn, 'r--', alpha=0.3, linewidth=1, label='NN Raw')
            axes[ax_idx].plot(timestamps, y_pred_nn_smooth, 'r-', linewidth=2, label='NN Smoothed')
        else:
            axes[ax_idx].plot(timestamps, y_pred_nn, 'r-', linewidth=2, label='NN Prediction')

        coverage = nn_metrics.get('coverage', 0)
        metrics_text = f"R²={nn_metrics['r2']:.3f}, RMSE={nn_metrics['rmse']:.2f}kW, Cov={coverage}"
        axes[ax_idx].set_title(f'Neural Network (Standard Prediction) | {metrics_text}', fontsize=12, fontweight='bold')
        axes[ax_idx].set_xlabel('Date', fontsize=11)
        axes[ax_idx].set_ylabel('Power (kW)', fontsize=11)
        axes[ax_idx].legend(loc='upper left', fontsize=10)
        axes[ax_idx].grid(True, alpha=0.3)
        axes[ax_idx].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axes[ax_idx].xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(timestamps) // 2880 // 10)))

    plt.tight_layout()
    plt.savefig('energy_prediction_overview.png', dpi=150, bbox_inches='tight')
    print("\n📊 Saved: energy_prediction_overview.png")
    plt.close()


def plot_forecast_48h(df_test, y_test, y_pred_lr, y_pred_nn, lr_metrics, nn_metrics,
                      y_pred_lr_smooth=None, y_pred_nn_smooth=None,
                      quantile=0.5, quantile_config=None):
    """
    Create detailed plot showing first 48 hours of predictions.
    Handles both single and dual model configurations.
    """
    n_hours = min(48, len(df_test))
    timestamps = df_test['timestamp'].iloc[:n_hours].values
    y_test_48h = y_test[:n_hours]

    # Determine which models we have
    has_qr = y_pred_lr is not None
    has_nn = y_pred_nn is not None

    if has_qr and has_nn:
        # Both models - use 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        axes = [ax1, ax2]
    else:
        # Single model - use 1 subplot
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        axes = [ax]

    fig.suptitle('Energy Load Prediction - 48-Hour Forecast Detail', fontsize=16, fontweight='bold')

    # Plot Quantile Regression if available
    if has_qr:
        ax_idx = 0 if has_qr and has_nn else 0
        y_pred_lr_48h = y_pred_lr[:n_hours]

        axes[ax_idx].plot(timestamps, y_test_48h, 'ko-', alpha=0.7, linewidth=2, markersize=4, label='Actual')
        if y_pred_lr_smooth is not None:
            y_pred_lr_smooth_48h = y_pred_lr_smooth[:n_hours]
            axes[ax_idx].plot(timestamps, y_pred_lr_48h, 'b--', alpha=0.4, linewidth=1.5, markersize=3, label='QR Raw')
            axes[ax_idx].plot(timestamps, y_pred_lr_smooth_48h, 'b-', linewidth=2.5, markersize=5, label='QR Smoothed', marker='o')
        else:
            axes[ax_idx].plot(timestamps, y_pred_lr_48h, 'bo-', linewidth=2.5, markersize=5, label='QR Prediction')

        if quantile_config is not None:
            peak_start = quantile_config['peak_start']
            peak_end = quantile_config['peak_end']
            peak_q = quantile_config['peak_quantile']
            offpeak_q = quantile_config['offpeak_quantile']
            title_q = f'Dynamic Q (Peak: {peak_q * 100:.0f}%, Off: {offpeak_q * 100:.0f}%)'
        else:
            title_q = f'{quantile * 100:.0f}th Percentile'

        coverage = lr_metrics.get('coverage', 0)
        metrics_text = f"R²={lr_metrics['r2']:.3f}, RMSE={lr_metrics['rmse']:.2f}kW, Cov={coverage}"
        axes[ax_idx].set_title(f'Quantile Regression ({title_q}) | {metrics_text}', fontsize=12, fontweight='bold')
        axes[ax_idx].set_ylabel('Power (kW)', fontsize=11)
        axes[ax_idx].legend(loc='upper left', fontsize=10)
        axes[ax_idx].grid(True, alpha=0.3)
        axes[ax_idx].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        if not has_nn:
            axes[ax_idx].set_xlabel('Date & Time', fontsize=11)

    # Plot Neural Network if available
    if has_nn:
        ax_idx = 1 if has_qr and has_nn else 0
        y_pred_nn_48h = y_pred_nn[:n_hours]

        axes[ax_idx].plot(timestamps, y_test_48h, 'ko-', alpha=0.7, linewidth=2, markersize=4, label='Actual')
        if y_pred_nn_smooth is not None:
            y_pred_nn_smooth_48h = y_pred_nn_smooth[:n_hours]
            axes[ax_idx].plot(timestamps, y_pred_nn_48h, 'r--', alpha=0.4, linewidth=1.5, markersize=3, label='NN Raw')
            axes[ax_idx].plot(timestamps, y_pred_nn_smooth_48h, 'r-', linewidth=2.5, markersize=5, label='NN Smoothed', marker='o')
        else:
            axes[ax_idx].plot(timestamps, y_pred_nn_48h, 'ro-', linewidth=2.5, markersize=5, label='NN Prediction')

        coverage = nn_metrics.get('coverage', 0)
        metrics_text = f"R²={nn_metrics['r2']:.3f}, RMSE={nn_metrics['rmse']:.2f}kW, Cov={coverage}"
        axes[ax_idx].set_title(f'Neural Network | {metrics_text}', fontsize=12, fontweight='bold')
        axes[ax_idx].set_xlabel('Date & Time', fontsize=11)
        axes[ax_idx].set_ylabel('Power (kW)', fontsize=11)
        axes[ax_idx].legend(loc='upper left', fontsize=10)
        axes[ax_idx].grid(True, alpha=0.3)
        axes[ax_idx].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

    plt.tight_layout()
    plt.savefig('energy_prediction_48h.png', dpi=150, bbox_inches='tight')
    print("📊 Saved: energy_prediction_48h.png")
    plt.close()


def print_summary(qr_metrics, nn_metrics, features, n_train, n_test,
                  is_iterative=False, quantile=0.5, quantile_config=None,
                  qr_coverage=0, nn_coverage=0, buffer_pct=0,
                  model_type='both'):
    """
    Print comprehensive summary of model performance.
    """
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 70)

    print(f"\n📊 Dataset:")
    print(f"   Training samples: {n_train}")
    print(f"   Testing samples:  {n_test}")
    print(f"   Features used: {len(features)}")
    if is_iterative:
        print(f"   Prediction mode: Iterative (using predicted power lags)")
    else:
        print(f"   Prediction mode: Standard")

    if quantile_config is not None:
        peak_start = quantile_config['peak_start']
        peak_end = quantile_config['peak_end']
        peak_q = quantile_config['peak_quantile']
        offpeak_q = quantile_config['offpeak_quantile']
        print(f"   Quantile mode: Dynamic")
        print(f"      Peak ({peak_start}:00-{peak_end}:00): {peak_q * 100:.0f}th percentile")
        print(f"      Off-peak: {offpeak_q * 100:.0f}th percentile")
    else:
        print(f"   Quantile: {quantile * 100:.0f}th percentile")

    if buffer_pct > 0:
        print(f"   Buffer: {buffer_pct:.1f}%")

    if model_type in ['quantile', 'both'] and qr_metrics:
        print(f"\n🎯 Quantile Regression:")
        print(f"   R² Score:  {qr_metrics['r2']:.4f}")
        print(f"   MAE:       {qr_metrics['mae']:.4f} kW")
        print(f"   RMSE:      {qr_metrics['rmse']:.4f} kW")
        print(f"   Coverage:  {qr_coverage:.1f}%")

    if model_type in ['neural', 'both'] and nn_metrics:
        print(f"\n🧠 Neural Network:")
        print(f"   R² Score:  {nn_metrics['r2']:.4f}")
        print(f"   MAE:       {nn_metrics['mae']:.4f} kW")
        print(f"   RMSE:      {nn_metrics['rmse']:.4f} kW")
        print(f"   Coverage:  {nn_coverage:.1f}%")