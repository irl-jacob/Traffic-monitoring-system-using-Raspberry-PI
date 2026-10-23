"""Unit tests for LineCounter and vector segment intersection."""

from src.line_counter import LineCounter, segments_intersect


def test_segments_intersect():
    # Perpendicular intersection
    assert segments_intersect((0, 5), (10, 5), (5, 0), (5, 10)) is True

    # Diagonal intersection
    assert segments_intersect((0, 0), (10, 10), (0, 10), (10, 0)) is True

    # Parallel segments
    assert segments_intersect((0, 0), (10, 0), (0, 5), (10, 5)) is False

    # Non-intersecting segments in general position
    assert segments_intersect((0, 0), (2, 2), (5, 5), (7, 7)) is False


def test_line_counter_single_crossing():
    # Horizontal line from (0, 50) to (100, 50)
    counter = LineCounter(p1=(0, 50), p2=(100, 50))

    # Frame 1: Vehicle 1 is above the line at (50, 20)
    evt1 = counter.update(track_id=1, class_name="car", current_centroid=(50.0, 20.0), frame_idx=1)
    assert evt1 is None
    assert counter.total_count == 0

    # Frame 2: Vehicle 1 crosses below the line to (50, 70)
    evt2 = counter.update(track_id=1, class_name="car", current_centroid=(50.0, 70.0), frame_idx=2)
    assert evt2 is not None
    assert evt2.track_id == 1
    assert evt2.class_name == "car"
    assert counter.total_count == 1
    assert counter.class_counts["car"] == 1

    # Frame 3: Vehicle 1 continues to (50, 90) -> FR4: must NOT count again
    evt3 = counter.update(track_id=1, class_name="car", current_centroid=(50.0, 90.0), frame_idx=3)
    assert evt3 is None
    assert counter.total_count == 1

    # Frame 4: Vehicle 1 briefly flickers back and forth across line -> still NOT counted again
    evt4 = counter.update(track_id=1, class_name="car", current_centroid=(50.0, 30.0), frame_idx=4)
    assert evt4 is None
    assert counter.total_count == 1


def test_line_counter_per_class_and_directions():
    counter = LineCounter(p1=(0, 50), p2=(100, 50))

    # Car moving downwards (20 -> 80)
    counter.update(track_id=10, class_name="car", current_centroid=(30.0, 20.0), frame_idx=1)
    evt_car = counter.update(track_id=10, class_name="car", current_centroid=(30.0, 80.0), frame_idx=2)
    assert evt_car is not None
    dir_car = evt_car.direction

    # Motorcycle moving upwards (80 -> 20)
    counter.update(track_id=20, class_name="motorcycle", current_centroid=(70.0, 80.0), frame_idx=1)
    evt_moto = counter.update(track_id=20, class_name="motorcycle", current_centroid=(70.0, 20.0), frame_idx=2)
    assert evt_moto is not None
    dir_moto = evt_moto.direction

    # Ensure opposite directions
    assert dir_car != dir_moto
    assert counter.total_count == 2
    assert counter.class_counts["car"] == 1
    assert counter.class_counts["motorcycle"] == 1
    assert counter.direction_counts["In"] == 1
    assert counter.direction_counts["Out"] == 1


def test_line_counter_cleanup_old_tracks():
    counter = LineCounter(p1=(0, 50), p2=(100, 50), max_missing_frames=10)
    counter.update(track_id=1, class_name="car", current_centroid=(10.0, 10.0), frame_idx=5)
    assert 1 in counter.track_history

    # Clean up at frame 20 (gap = 15 > 10)
    counter.cleanup_old_tracks(current_frame_idx=20)
    assert 1 not in counter.track_history
