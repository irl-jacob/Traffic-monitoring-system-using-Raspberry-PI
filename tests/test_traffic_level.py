"""Unit tests for TrafficLevelEngine, rolling flow, density, and hysteresis."""

from src.traffic_level import TrafficLevel, TrafficLevelEngine, is_point_in_polygon


def test_point_in_polygon():
    # Square from (0, 0) to (10, 10)
    poly = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

    # Point inside
    assert is_point_in_polygon((5.0, 5.0), poly) is True

    # Point clearly outside
    assert is_point_in_polygon((15.0, 5.0), poly) is False
    assert is_point_in_polygon((5.0, -2.0), poly) is False


def test_traffic_level_flow_classification():
    engine = TrafficLevelEngine(
        flow_window_sec=30.0,
        hysteresis_sec=0.0,  # Disable hysteresis for isolated threshold testing
        flow_thresholds={"low_max": 8.0, "medium_max": 20.0},
        density_thresholds={"low_max": 3, "medium_max": 8}
    )

    assert engine.classify_flow(5.0) == TrafficLevel.LOW
    assert engine.classify_flow(8.0) == TrafficLevel.MEDIUM
    assert engine.classify_flow(15.0) == TrafficLevel.MEDIUM
    assert engine.classify_flow(20.0) == TrafficLevel.MEDIUM
    assert engine.classify_flow(25.0) == TrafficLevel.HIGH


def test_traffic_level_density_classification():
    engine = TrafficLevelEngine(
        flow_window_sec=30.0,
        hysteresis_sec=0.0,
        flow_thresholds={"low_max": 8.0, "medium_max": 20.0},
        density_thresholds={"low_max": 3, "medium_max": 8}
    )

    assert engine.classify_density(2) == TrafficLevel.LOW
    assert engine.classify_density(3) == TrafficLevel.LOW
    assert engine.classify_density(4) == TrafficLevel.MEDIUM
    assert engine.classify_density(8) == TrafficLevel.MEDIUM
    assert engine.classify_density(10) == TrafficLevel.HIGH


def test_hysteresis_state_machine():
    engine = TrafficLevelEngine(
        flow_window_sec=30.0,
        hysteresis_sec=2.0,
        flow_thresholds={"low_max": 8.0, "medium_max": 20.0},
        density_thresholds={"low_max": 3, "medium_max": 8},
        roi_polygon=[(0, 0), (100, 0), (100, 100), (0, 100)]
    )

    # At t=0s, 0 vehicles -> LOW
    lvl, flow, density = engine.update(current_time=0.0, active_centroids=[])
    assert lvl == TrafficLevel.LOW

    # At t=1.0s, suddenly 10 vehicles in ROI (qualifies as HIGH)
    high_density_centroids = [(10 + i * 5, 20) for i in range(10)]
    lvl, flow, density = engine.update(current_time=1.0, active_centroids=high_density_centroids)
    # Hysteresis requirement: only 1.0s has passed since candidate became HIGH (< 2.0s), so still LOW
    assert lvl == TrafficLevel.LOW
    assert engine.candidate_level == TrafficLevel.HIGH

    # At t=2.5s (1.5s into candidate, still < 2.0s)
    lvl, _, _ = engine.update(current_time=2.5, active_centroids=high_density_centroids)
    assert lvl == TrafficLevel.LOW

    # At t=3.1s (2.1s into candidate >= 2.0s debounce time)
    lvl, _, _ = engine.update(current_time=3.1, active_centroids=high_density_centroids)
    # Level should now successfully switch to HIGH!
    assert lvl == TrafficLevel.HIGH
    assert engine.reported_level == TrafficLevel.HIGH
    assert engine.peak_level == TrafficLevel.HIGH


def test_summary_metrics_generation():
    engine = TrafficLevelEngine(flow_window_sec=10.0, hysteresis_sec=1.0)
    engine.update(current_time=0.0, active_centroids=[])
    engine.update(current_time=5.0, active_centroids=[])
    engine.finalize(end_time=5.0)

    summary = engine.get_summary_metrics()
    assert summary["peak_level"] == "LOW"
    assert "time_spent_in_levels_sec" in summary
    assert summary["total_monitoring_duration_sec"] == 5.0
