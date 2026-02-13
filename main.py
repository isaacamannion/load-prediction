#!/usr/bin/env python3
"""
Energy Load Prediction System

Predicts household energy consumption using Home Assistant historical data.
Supports both Quantile Regression (for conservative forecasting) and Neural Networks.
"""

import argparse
import numpy as np
from sklearn.model_selection import train_test_split

from config import DataConfig, ModelConfig, PostProcessConfig, get_recommended_config
from data_processing import parse_ha_export, add_lagged_features, apply_smoothing, get_default_features
from models import QuantileRegressionModel, NeuralNetworkModel, predict_iterative
from visualization import plot_overview, plot_forecast_48h, print_summary


def main():
    """Main entry point for the energy load prediction system."""

    parser = argparse.ArgumentParser(
        description='Energy Load Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --csv data.csv --model quantile
  python main.py --csv data.csv --model both --quantile 0.75
  python main.py --csv data.csv --recommended
        """
    )

    parser.add_argument('--csv', help='Path to Home Assistant CSV export')
    parser.add_argument('--model', choices=['quantile', 'neural', 'both'],
                        default='quantile', help='Model type to use')
    parser.add_argument('--bins', type=int, default=60,
                        help='Time bin size in minutes (default: 60)')
    parser.add_argument('--power-lags', type=int, default=10,
                        help='Number of power lag features (default: 10)')
    parser.add_argument('--temp-lags', type=int, default=5,
                        help='Number of temperature lag features (default: 5)')
    parser.add_argument('--quantile', type=float, default=0.75,
                        help='Target quantile for predictions (0.5-0.99, default: 0.75)')
    parser.add_argument('--dynamic', action='store_true',
                        help='Use dynamic quantile (different for peak/off-peak)')
    parser.add_argument('--peak-start', type=int, default=9,
                        help='Peak period start hour (default: 9)')
    parser.add_argument('--peak-end', type=int, default=22,
                        help='Peak period end hour (default: 22)')
    parser.add_argument('--buffer', type=float, default=0.0,
                        help='Safety buffer percentage (default: 0)')
    parser.add_argument('--smooth', action='store_true',
                        help='Apply smoothing to predictions')
    parser.add_argument('--recommended', action='store_true',
                        help='Use recommended configuration')

    args = parser.parse_args()

    # Check if running in interactive mode (no CSV file provided)
    if args.csv is None:
        print("\n" + "=" * 70)
        print("ENERGY LOAD PREDICTION SYSTEM - INTERACTIVE MODE")
        print("=" * 70)
        print("\nNo command-line arguments detected. Starting interactive setup...")

        # Ask if user wants recommended settings
        print("\n" + "-" * 70)
        print("CONFIGURATION MODE")
        print("-" * 70)
        print("1. Use recommended settings (easiest - good results)")
        print("2. Custom configuration (advanced)")

        mode_choice = input("\nYour choice (1 or 2, default=1): ").strip()

        # Get CSV file path
        csv_file = input("\nPath to CSV file (e.g., HA_history_export.csv): ").strip()
        while not csv_file:
            print("  ⚠️  CSV file path is required!")
            csv_file = input("Path to CSV file: ").strip()

        if mode_choice == '2':
            # Custom configuration
            print("\n" + "-" * 70)
            print("MODEL SELECTION")
            print("-" * 70)
            print("1. Quantile Regression (conservative, recommended)")
            print("2. Neural Network (standard regression)")
            print("3. Both (for comparison)")

            model_choice = input("\nYour choice (1-3, default=1): ").strip()
            model_map = {'1': 'quantile', '2': 'neural', '3': 'both', '': 'quantile'}
            model_type = model_map.get(model_choice, 'quantile')

            print("\n" + "-" * 70)
            print("DATA CONFIGURATION")
            print("-" * 70)

            bins_input = input("Time bin size in minutes (default=60): ").strip()
            bins = int(bins_input) if bins_input.isdigit() else 60

            power_lags_input = input("Number of power lag features (default=10): ").strip()
            power_lags = int(power_lags_input) if power_lags_input.isdigit() else 10

            temp_lags_input = input("Number of temperature lag features (default=5): ").strip()
            temp_lags = int(temp_lags_input) if temp_lags_input.isdigit() else 5

            print("\n" + "-" * 70)
            print("QUANTILE CONFIGURATION")
            print("-" * 70)
            print("Use dynamic quantile? (different targets for peak/off-peak hours)")
            dynamic_input = input("Dynamic quantile? (y/n, default=y): ").strip().lower()
            use_dynamic = dynamic_input != 'n'

            if use_dynamic:
                peak_start_input = input("Peak period start hour (0-23, default=9): ").strip()
                peak_start = int(peak_start_input) if peak_start_input.isdigit() else 9

                peak_end_input = input("Peak period end hour (0-23, default=22): ").strip()
                peak_end = int(peak_end_input) if peak_end_input.isdigit() else 22

                quantile_input = input("Peak quantile (0.5-0.99, default=0.75): ").strip()
                quantile = float(quantile_input) if quantile_input else 0.75
            else:
                peak_start = 9
                peak_end = 22
                quantile_input = input("Target quantile (0.5-0.99, default=0.75): ").strip()
                quantile = float(quantile_input) if quantile_input else 0.75

            print("\n" + "-" * 70)
            print("POST-PROCESSING")
            print("-" * 70)

            buffer_input = input("Safety buffer percentage (0-50, default=0): ").strip()
            buffer = float(buffer_input) if buffer_input else 0.0

            smooth_input = input("Apply smoothing? (y/n, default=n): ").strip().lower()
            apply_smooth = smooth_input == 'y'

            # Create configurations
            data_config = DataConfig(
                csv_file=csv_file,
                bin_size_minutes=bins,
                n_power_lags=power_lags,
                n_temp_lags=temp_lags
            )

            model_config = ModelConfig(
                model_type=model_type,
                quantile=quantile,
                use_dynamic_quantile=use_dynamic,
                peak_start=peak_start,
                peak_end=peak_end
            )

            post_config = PostProcessConfig(
                buffer_percentage=buffer,
                apply_smoothing=apply_smooth
            )
        else:
            # Recommended settings
            print("\n✓ Using recommended configuration")
            data_config, model_config, post_config = get_recommended_config()
            data_config.csv_file = csv_file
    else:
        # Command-line mode
        if args.recommended:
            print("\n🎯 Using recommended configuration...")
            data_config, model_config, post_config = get_recommended_config()
            data_config.csv_file = args.csv
        else:
            data_config = DataConfig(
                csv_file=args.csv,
                bin_size_minutes=args.bins,
                n_power_lags=args.power_lags,
                n_temp_lags=args.temp_lags
            )

            model_config = ModelConfig(
                model_type=args.model,
                quantile=args.quantile,
                use_dynamic_quantile=args.dynamic,
                peak_start=args.peak_start,
                peak_end=args.peak_end
            )

            post_config = PostProcessConfig(
                buffer_percentage=args.buffer,
                apply_smoothing=args.smooth
            )

    # Parse and prepare data
    print("\n" + "=" * 70)
    print("ENERGY LOAD PREDICTION SYSTEM")
    print("=" * 70)

    df = parse_ha_export(data_config.csv_file, data_config.bin_size_minutes)
    df = add_lagged_features(df, data_config.n_power_lags, data_config.n_temp_lags)

    # Select features
    features = get_default_features()
    features = [f for f in features if f in df.columns]

    for i in range(1, data_config.n_power_lags + 1):
        features.append(f'power_lag_{i}')
    for i in range(1, data_config.n_temp_lags + 1):
        features.append(f'temp_lag_{i}')

    print(f"\n✓ Using {len(features)} features: {', '.join(features)}")

    # Split data
    X = df[features].values
    y = df['consumption'].values

    test_size = (100 - data_config.train_percentage) / 100.0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    train_size = len(X_train)
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]

    hours_train = df_train['hour'].values
    hours_test = df_test['hour'].values

    print(f"\n📊 Training: {len(X_train)} samples | Testing: {len(X_test)} samples")

    # Initialize results
    results = {}
    quantile_config = model_config.get_dynamic_config()

    # Train Quantile Regression if requested
    if model_config.model_type in ['quantile', 'both']:
        print("\n" + "=" * 70)
        print("🎯 QUANTILE REGRESSION MODEL")
        print("=" * 70)

        qr_model = QuantileRegressionModel(model_config.quantile, quantile_config)
        qr_model.train(X_train, y_train, hours_train)

        print("   Making predictions...")
        qr_pred_result = predict_iterative(
            X_test, y_test, qr_model, features,
            data_config.n_power_lags, hours_test
        )

        y_pred_qr = qr_pred_result['predictions']
        qr_metrics = qr_model.evaluate(y_test, y_pred_qr)
        qr_coverage = qr_model.calculate_coverage(y_test, y_pred_qr)

        print(f"   ✓ Coverage: {qr_coverage:.1f}% of actuals below predictions")

        results['qr_model'] = qr_model
        results['y_pred_qr'] = y_pred_qr
        results['qr_metrics'] = qr_metrics
        results['qr_coverage'] = qr_coverage
        results['is_iterative'] = qr_pred_result['iterative']
    else:
        results['qr_model'] = None
        results['y_pred_qr'] = None
        results['qr_metrics'] = None
        results['qr_coverage'] = 0
        results['is_iterative'] = False

    # Train Neural Network if requested
    if model_config.model_type in ['neural', 'both']:
        print("\n" + "=" * 70)
        print("🧠 NEURAL NETWORK MODEL")
        print("=" * 70)

        nn_model = NeuralNetworkModel()
        nn_model.train(X_train, y_train)

        print("   Making predictions...")
        nn_pred_result = predict_iterative(
            X_test, y_test, nn_model, features,
            data_config.n_power_lags
        )

        y_pred_nn = nn_pred_result['predictions']
        nn_metrics = nn_model.evaluate(y_test, y_pred_nn)
        nn_coverage = nn_model.calculate_coverage(y_test, y_pred_nn)

        print(f"   ✓ Coverage: {nn_coverage:.1f}% of actuals below predictions")

        results['nn_model'] = nn_model
        results['y_pred_nn'] = y_pred_nn
        results['nn_metrics'] = nn_metrics
        results['nn_coverage'] = nn_coverage
    else:
        results['nn_model'] = None
        results['y_pred_nn'] = None
        results['nn_metrics'] = None
        results['nn_coverage'] = 0

    # Apply buffer if requested
    if post_config.buffer_percentage > 0:
        print("\n" + "=" * 70)
        print(f"📊 APPLYING {post_config.buffer_percentage:.1f}% SAFETY BUFFER")
        print("=" * 70)

        buffer_mult = 1.0 + (post_config.buffer_percentage / 100.0)

        if results['y_pred_qr'] is not None:
            results['y_pred_qr'] = results['y_pred_qr'] * buffer_mult
            results['qr_metrics'] = results['qr_model'].evaluate(y_test, results['y_pred_qr'])
            results['qr_coverage'] = results['qr_model'].calculate_coverage(y_test, results['y_pred_qr'])
            print(f"   QR Coverage: {results['qr_coverage']:.1f}%")

        if results['y_pred_nn'] is not None:
            results['y_pred_nn'] = results['y_pred_nn'] * buffer_mult
            results['nn_metrics'] = results['nn_model'].evaluate(y_test, results['y_pred_nn'])
            results['nn_coverage'] = results['nn_model'].calculate_coverage(y_test, results['y_pred_nn'])
            print(f"   NN Coverage: {results['nn_coverage']:.1f}%")

    # Apply smoothing if requested
    if post_config.apply_smoothing:
        print("\n" + "=" * 70)
        print(f"✨ APPLYING {post_config.smoothing_method.upper()} SMOOTHING")
        print(f"   Window size: {post_config.smoothing_window}")
        print("=" * 70)

        if results['y_pred_qr'] is not None:
            results['y_pred_qr_smooth'] = apply_smoothing(
                results['y_pred_qr'],
                post_config.smoothing_window,
                post_config.smoothing_method
            )
            print("   ✓ QR smoothing applied")

        if results['y_pred_nn'] is not None:
            results['y_pred_nn_smooth'] = apply_smoothing(
                results['y_pred_nn'],
                post_config.smoothing_window,
                post_config.smoothing_method
            )
            print("   ✓ NN smoothing applied")

    # Store results
    results['df_test'] = df_test
    results['y_test'] = y_test
    results['features'] = features
    results['quantile'] = model_config.quantile
    results['quantile_config'] = quantile_config
    results['model_type'] = model_config.model_type

    # Print summary
    print_summary(
        results['qr_metrics'],
        results['nn_metrics'],
        results['features'],
        len(X_train),
        len(X_test),
        results['is_iterative'],
        results['quantile'],
        results['quantile_config'],
        results['qr_coverage'],
        results['nn_coverage'],
        post_config.buffer_percentage,
        model_config.model_type
    )

    # Generate visualizations
    if model_config.model_type == 'both':
        print("\n" + "=" * 70)
        print("📊 GENERATING VISUALIZATIONS")
        print("=" * 70)

        qr_metrics_viz = {**results['qr_metrics'], 'coverage': f"{results['qr_coverage']:.1f}%"}
        nn_metrics_viz = {**results['nn_metrics'], 'coverage': f"{results['nn_coverage']:.1f}%"}

        plot_overview(
            results['df_test'], results['y_test'],
            results['y_pred_qr'], results['y_pred_nn'],
            qr_metrics_viz, nn_metrics_viz,
            results['features'],
            results.get('y_pred_qr_smooth'),
            results.get('y_pred_nn_smooth'),
            is_iterative=results['is_iterative'],
            quantile=results['quantile'],
            quantile_config=results['quantile_config']
        )

        plot_forecast_48h(
            results['df_test'], results['y_test'],
            results['y_pred_qr'], results['y_pred_nn'],
            qr_metrics_viz, nn_metrics_viz,
            results.get('y_pred_qr_smooth'),
            results.get('y_pred_nn_smooth'),
            results['quantile'],
            results['quantile_config']
        )
    elif model_config.model_type == 'quantile' and results['y_pred_qr'] is not None:
        print("\n" + "=" * 70)
        print("📊 GENERATING VISUALIZATIONS")
        print("=" * 70)

        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        qr_metrics_viz = {**results['qr_metrics'], 'coverage': f"{results['qr_coverage']:.1f}%"}

        # Create overview plot
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        timestamps = df_test['timestamp'].values

        ax.plot(timestamps, y_test, 'k-', alpha=0.6, linewidth=1.5, label='Actual')
        ax.plot(timestamps, results['y_pred_qr'], 'b-', linewidth=2, label='QR Prediction')

        if quantile_config:
            title_q = f"Dynamic Q (Peak: {quantile_config['peak_quantile'] * 100:.0f}%, Off: {quantile_config['offpeak_quantile'] * 100:.0f}%)"
        else:
            title_q = f"{model_config.quantile * 100:.0f}th Percentile"

        ax.set_title(
            f'Quantile Regression ({title_q}) | R²={qr_metrics_viz["r2"]:.3f}, Coverage={qr_metrics_viz["coverage"]}',
            fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Power (kW)', fontsize=11)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        plt.tight_layout()
        plt.savefig('energy_prediction_overview.png', dpi=150, bbox_inches='tight')
        print("\n📊 Saved: energy_prediction_overview.png")
        plt.close()

    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()