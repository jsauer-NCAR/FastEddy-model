"""Convert FastEddy rank-wise binary outputs to netCDF while the simulation runs.

Runs alongside a FastEddy job (ioOutputMode = 1). Rank 0 polls the binary output
directory; an output timestep is READY when all numOutRanks rank files exist, their
sizes did not change since the previous poll, and the newest of them is at least
--settle seconds old. Ready timesteps are converted (FEbinaryToNetCDF.convert_timestep)
into <netCDFpath>/<FEoutBase>.<step>, written atomically, and their binaries deleted.

Batching: with N MPI ranks, conversion waits until --batch ready timesteps exist
(default N) and then spreads them round-robin over the ranks. A partial batch is
converted anyway once its oldest member has waited --max-wait seconds, when the
final timestep (--tend) is ready, or once the simulation has ended (--stop-file).
Between polls every rank sleeps (no busy waiting in MPI collectives).

The run ends successfully when every expected timestep tstart..tend (step tstep) has
a netCDF file. If the stop file appears (the FastEddy job ended, successfully or not),
remaining ready timesteps are converted and the run exits non-zero if any expected
timestep is still missing, so dependent jobs do not start on incomplete output.

Usage (one converter per FastEddy job):
  mpiexec -n 8 python -u concurrent_converter.py -f convert.json -a field_attributes.json \
      --tend 576000 --stop-file <run dir>/.fe_B1.ended [--batch 8] [--poll 30] [--settle 20]

convert.json is the same file FEbinaryToNetCDF.py reads (outpath, FEoutBase, numOutRanks,
tstart, tstep, netCDFpath, removeBinaries); fileSetSize is ignored.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

from mpi4py import MPI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import FEbinaryToNetCDF as fe2nc  # noqa: E402  (reuses its reader, attributes and writer)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-f", "--file", required=True, help="JSON converter parameters (FEbinaryToNetCDF format)")
    p.add_argument("-a", "--attrs", required=True, help="JSON field attribute definitions")
    p.add_argument("--tend", type=int, required=True, help="Last output timestep this converter must produce")
    p.add_argument("--stop-file", default=None, help="File created when the FastEddy job ends (success or failure)")
    p.add_argument("--batch", type=int, default=None, help="Ready timesteps to collect before converting (default: MPI size)")
    p.add_argument("--poll", type=float, default=30.0, help="Seconds between directory scans")
    p.add_argument("--settle", type=float, default=20.0, help="Seconds a timestep's newest rank file must be untouched")
    p.add_argument("--max-wait", type=float, default=900.0, help="Convert a partial batch after this many seconds")
    p.add_argument("--start-timeout", type=float, default=6 * 3600.0,
                   help="Give up if neither output nor the stop file appears within this many seconds")
    return p.parse_args()


def log(msg):
    print(f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} [rank {MPI.COMM_WORLD.Get_rank()}] {msg}", flush=True)


class Scanner:
    """Rank-0 bookkeeping of which output timesteps are complete on disk."""

    def __init__(self, p, tend, settle):
        self.p, self.tend, self.settle = p, tend, settle
        self.expected = list(range(p["tstart"], tend + 1, p["tstep"]))
        self.prev_sizes = {}      # step -> tuple of rank-file sizes at the previous scan
        self.first_ready = {}     # step -> time it was first seen ready

    def nc_path(self, step):
        return os.path.join(self.p["netCDFpath"], f"{self.p['FEoutBase']}.{step}")

    def done(self, step):
        return os.path.exists(self.nc_path(step))

    def remaining(self):
        return [s for s in self.expected if not self.done(s)]

    def pending_binaries(self):
        """Expected steps that still have binaries on disk (e.g. a segment's rewritten start step)."""
        return [s for s in self.expected if os.path.exists(os.path.join(
            self.p["outpath"], fe2nc.binary_files_for(self.p["FEoutBase"], 1, s)[0]))]

    def ready(self, now):
        out = []
        for step in self.expected:
            files = [os.path.join(self.p["outpath"], f)
                     for f in fe2nc.binary_files_for(self.p["FEoutBase"], self.p["numOutRanks"], step)]
            try:
                stats = [os.stat(f) for f in files]
            except FileNotFoundError:
                self.prev_sizes.pop(step, None)
                continue
            sizes = tuple(s.st_size for s in stats)
            stable = self.prev_sizes.get(step) == sizes
            settled = now - max(s.st_mtime for s in stats) >= self.settle
            self.prev_sizes[step] = sizes
            if stable and settled and min(sizes) > 0:
                self.first_ready.setdefault(step, now)
                out.append(step)
        return out


def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    args = parse_args()
    with open(args.file) as f:
        params = json.load(f)
    batch = args.batch or size
    attr_tables = fe2nc.load_field_attributes(args.attrs)
    if rank == 0:
        os.makedirs(params["netCDFpath"], exist_ok=True)
        log(f"watching {params['outpath']}{params['FEoutBase']}_rank_*.<step> for steps "
            f"{params['tstart']}..{args.tend} every {params['tstep']}; {size} rank(s), batch {batch}, "
            f"poll {args.poll}s, settle {args.settle}s, removeBinaries={params['removeBinaries']}")
    scan = Scanner(params, args.tend, args.settle) if rank == 0 else None
    t0 = time.time()
    converted = 0

    while True:
        plan = None
        if rank == 0:
            now = time.time()
            ended = bool(args.stop_file) and os.path.exists(args.stop_file)
            remaining = scan.remaining()
            ready = scan.ready(now)
            leftovers = scan.pending_binaries() if params["removeBinaries"] else []
            flush = (ended or not remaining or args.tend in ready
                     or (ready and now - min(scan.first_ready[s] for s in ready) >= args.max_wait))
            if not remaining and not leftovers:
                plan = {"action": "finish", "ok": True}
            elif ready and (len(ready) >= batch or flush):
                plan = {"action": "convert", "steps": ready}
            elif ended and not ready:
                # the job ended: allow two settle periods for its last writes, then stop
                if now - os.path.getmtime(args.stop_file) < 2 * args.settle:
                    plan = {"action": "sleep"}
                else:
                    plan = {"action": "finish", "ok": not remaining, "missing": remaining}
            elif not ended and not scan.prev_sizes and converted == 0 and now - t0 > args.start_timeout:
                plan = {"action": "finish", "ok": False, "missing": remaining, "why": "no output appeared"}
            elif not remaining and leftovers and not ready:
                plan = {"action": "sleep"}  # rewritten binaries still settling
            else:
                plan = {"action": "sleep"}
        plan = comm.bcast(plan, root=0)

        if plan["action"] == "finish":
            if rank == 0:
                if plan["ok"]:
                    log(f"all {len(scan.expected)} timesteps through {args.tend} are converted; done")
                else:
                    miss = plan["missing"]
                    log(f"stopping with {len(miss)} timesteps not converted "
                        f"({plan.get('why', 'simulation ended')}): {miss[:5]}{' ...' if len(miss) > 5 else ''}")
            comm.Barrier()
            return 0 if plan["ok"] else 1

        if plan["action"] == "sleep":
            time.sleep(args.poll)
            continue

        steps = plan["steps"]
        mine = steps[rank::size]
        if rank == 0:
            log(f"converting {len(steps)} timestep(s) {steps[0]}..{steps[-1]} over {size} rank(s)")
        ok = 0
        for step in mine:
            t = time.time()
            if fe2nc.convert_timestep(step, params["outpath"], params["FEoutBase"], params["numOutRanks"],
                                      params["netCDFpath"], params["removeBinaries"], attr_tables):
                ok += 1
                log(f"step {step} converted in {time.time() - t:.0f}s")
        converted += comm.allreduce(ok, op=MPI.SUM)


if __name__ == "__main__":
    status = main()
    MPI.Finalize()
    sys.exit(status)
