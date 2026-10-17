import pytest

from scripts.kaggle_saved_slat_watch import classify_kernel_status


def test_kaggle_status_does_not_treat_queued_as_terminal():
    assert classify_kernel_status('x has status "KernelWorkerStatus.QUEUED"') == "pending"
    assert classify_kernel_status('x has status "KernelWorkerStatus.RUNNING"') == "pending"


def test_kaggle_status_terminal_and_unknown_fail_loud():
    assert classify_kernel_status('x has status "KernelWorkerStatus.COMPLETE"') == "complete"
    assert classify_kernel_status('x has status "KernelWorkerStatus.ERROR"') == "failed"
    with pytest.raises(ValueError, match="unrecognized"):
        classify_kernel_status("empty or stale response")
