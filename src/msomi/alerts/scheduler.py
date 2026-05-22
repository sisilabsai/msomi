"""APScheduler wrapper for automated Msomi scanning and reporting jobs."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class MsomiScheduler:
    """
    Wraps APScheduler's BackgroundScheduler to manage recurring Msomi jobs.

    Usage:
        scheduler = MsomiScheduler()
        scheduler.add_signal_scan(my_scan_fn, interval_minutes=60)
        scheduler.add_eod_report(my_report_fn)
        scheduler.start()
        # ... runs in background ...
        scheduler.stop()
    """

    def __init__(self) -> None:
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.executors.pool import ThreadPoolExecutor

            self._scheduler = BackgroundScheduler(
                executors={"default": ThreadPoolExecutor(max_workers=4)},
                job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 300},
                timezone="UTC",
            )
        except ImportError:
            logger.warning("APScheduler not installed — scheduling disabled")
            self._scheduler = None

        self._jobs: list[str] = []

    # ── Job Registration ──────────────────────────────────────────────────────

    def add_signal_scan(
        self,
        fn: Callable,
        interval_minutes: int = 60,
        job_id: str = "signal_scan",
    ) -> None:
        """Run signal scan every `interval_minutes` minutes."""
        if self._scheduler is None:
            return
        self._scheduler.add_job(
            fn,
            trigger="interval",
            minutes=interval_minutes,
            id=job_id,
            replace_existing=True,
        )
        self._jobs.append(job_id)
        logger.info("Scheduled signal scan every %d min [id=%s]", interval_minutes, job_id)

    def add_eod_report(
        self,
        fn: Callable,
        hour: int = 22,
        minute: int = 0,
        job_id: str = "eod_report",
    ) -> None:
        """Run EOD report at a fixed UTC time daily."""
        if self._scheduler is None:
            return
        self._scheduler.add_job(
            fn,
            trigger="cron",
            hour=hour,
            minute=minute,
            id=job_id,
            replace_existing=True,
        )
        self._jobs.append(job_id)
        logger.info("Scheduled EOD report at %02d:%02d UTC [id=%s]", hour, minute, job_id)

    def add_weekly_review(
        self,
        fn: Callable,
        day_of_week: str = "sun",
        hour: int = 20,
        minute: int = 0,
        job_id: str = "weekly_review",
    ) -> None:
        """Run weekly review on `day_of_week` at the specified UTC time."""
        if self._scheduler is None:
            return
        self._scheduler.add_job(
            fn,
            trigger="cron",
            day_of_week=day_of_week,
            hour=hour,
            minute=minute,
            id=job_id,
            replace_existing=True,
        )
        self._jobs.append(job_id)
        logger.info(
            "Scheduled weekly review on %s at %02d:%02d UTC [id=%s]",
            day_of_week, hour, minute, job_id,
        )

    def add_custom(
        self,
        fn: Callable,
        trigger: str,
        job_id: str,
        **trigger_kwargs,
    ) -> None:
        """Add a fully custom APScheduler job."""
        if self._scheduler is None:
            return
        self._scheduler.add_job(fn, trigger=trigger, id=job_id, replace_existing=True, **trigger_kwargs)
        self._jobs.append(job_id)
        logger.info("Scheduled custom job [id=%s] trigger=%s", job_id, trigger)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background scheduler."""
        if self._scheduler is None:
            logger.warning("Scheduler not available (APScheduler not installed)")
            return
        if not self._scheduler.running:
            self._scheduler.start()
            logger.info("Scheduler started with %d job(s): %s", len(self._jobs), self._jobs)

    def stop(self, wait: bool = True) -> None:
        """Gracefully stop the scheduler."""
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=wait)
            logger.info("Scheduler stopped")

    @property
    def is_running(self) -> bool:
        return self._scheduler is not None and self._scheduler.running

    def list_jobs(self) -> list[dict]:
        """Return a summary of all scheduled jobs."""
        if self._scheduler is None:
            return []
        return [
            {
                "id": job.id,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger),
            }
            for job in self._scheduler.get_jobs()
        ]
