"""Progress bars for Orion benchmark compatible with logging and dask."""

import io
import logging
from datetime import datetime, timedelta

LOGGER = logging.getLogger(__name__)


class TqdmLogger(io.StringIO):

    _buffer = ''

    def write(self, buf):
        self._buffer = buf.strip('\r\n\t ')

    def flush(self):
        LOGGER.info(self._buffer)


def progress(*futures):
    """Track progress of dask computation in a remote cluster.
    LogProgressBar is defined inside here to avoid having to import
    its dependencies if not used.
    """
    # Import distributed only when used
    from distributed.client import futures_of  # pylint: disable=C0415
    from distributed.diagnostics.progressbar import TextProgressBar  # pylint: disable=c0415

    class LogProgressBar(TextProgressBar):
        """Dask progress bar based on logging instead of stdout."""

        last = 0
        logger = logging.getLogger('distributed')

        def _draw_bar(self, remaining, all, **kwargs):   # pylint: disable=W0221,W0622
            done = all - remaining
            frac = (done / all) if all else 0

            if frac > self.last + 0.01:
                self.last = int(frac * 100) / 100
                bar = "#" * int(self.width * frac)
                percent = int(100 * frac)

                time_per_task = self.elapsed / (all - remaining)
                remaining_time = timedelta(seconds=time_per_task * remaining)
                eta = datetime.utcnow() + remaining_time

                elapsed = timedelta(seconds=self.elapsed)
                msg = "[{0:<{1}}] | {2}/{3} ({4}%) Completed | {5} | {6} | {7}".format(
                    bar, self.width, done, all, percent, elapsed, remaining_time, eta
                )
                self.logger.info(msg)
                LOGGER.info(msg)

        def _draw_stop(self, **kwargs):
            pass

    futures = futures_of(futures)
    if not isinstance(futures, (set, list)):
        futures = [futures]

    LogProgressBar(futures)
