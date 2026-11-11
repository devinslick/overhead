import sys
sys.path.insert(0, ".")
from overhead.tracker import IoUTracker

def test_birth_confirmation_and_death():
    tr = IoUTracker()
    # object appears at x=100, moves +8px/frame
    for f in range(10):
        x = 100 + 8 * f
        tr.step([[x, 300, x + 60, 360]])
    assert len(tr.tracks) == 1 and tr.tracks[0].confirmed
    tid = tr.tracks[0].id
    # detection drops out for 8 frames -> prediction coasts, same ID
    for _ in range(8):
        tr.step([])
    assert len(tr.tracks) == 1 and tr.tracks[0].id == tid
    # gone past max_misses -> reaped
    for _ in range(8):
        tr.step([])
    assert len(tr.tracks) == 0
    print(f"OK birth/coast/death (id={tid})")

def test_two_objects_no_swap():
    tr = IoUTracker()
    for f in range(30):
        # left object moves right, right object moves left; they pass near
        lx, rx = 100 + 4 * f, 900 - 4 * f
        tr.step([[lx, 300, lx + 50, 350], [rx, 300, rx + 50, 350]])
    assert len(tr.tracks) == 2
    ids = sorted(t.id for t in tr.tracks)
    left = [t for t in tr.tracks if t.center[0] < 640][0]
    right = [t for t in tr.tracks if t.center[0] >= 640][0]
    assert left.id != right.id
    print(f"OK two objects tracked separately {ids}")

def test_velocity_prediction():
    tr = IoUTracker()
    for f in range(12):
        x = 200 + 10 * f
        tr.step([[x, 400, x + 40, 440]])
    before = tr.tracks[0].center[0]
    tr.step([])  # miss -> predicted position should advance ~10px
    after = tr.tracks[0].center[0]
    assert 5 < (after - before) < 15, (before, after)
    print(f"OK coast prediction advanced {after-before:.1f}px")

if __name__ == "__main__":
    test_birth_confirmation_and_death()
    test_two_objects_no_swap()
    test_velocity_prediction()
    print("ALL TRACKER TESTS PASS")
