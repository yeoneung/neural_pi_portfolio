"""Evaluate completed pairs without publishing a partial completion marker."""
import argparse
import csv
import time
from followup_continuous_boundary import initialize, simulation, write_csv, write_json
from followup_io import BOUNDARY_DEFAULT, project_path

def run(out, timeout=3600):
    cfg=initialize(out)
    rows=[]
    for seed in cfg['seeds']:
        required=[out/phase/f'seed{seed}'/mode/'completion.json'
                  for phase in ['frozen','iteration'] for mode in cfg['modes']]
        deadline=time.monotonic()+timeout
        while not all(p.exists() for p in required):
            if time.monotonic()>=deadline:
                raise TimeoutError(f'Training for seed {seed} is incomplete after {timeout:g} seconds.')
            time.sleep(min(2,max(0,deadline-time.monotonic())))
        report=out/'simulation_reports'/f'seed{seed}'
        simulation(out,{**cfg,'seeds':[seed]},report_dir=report)
        part=out/f'simulation_seed{seed}.csv'
        part.write_bytes((report/'simulation_summary.csv').read_bytes())
        with part.open(encoding='utf-8',newline='') as stream:
            rows.extend(csv.DictReader(stream))
    write_csv(out/'simulation_summary.csv',rows)
    write_json(out/'simulation_completion.json',dict(status='complete',rows=len(rows),seeds=cfg['seeds']))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=project_path,default=BOUNDARY_DEFAULT)
    parser.add_argument('--timeout',type=float,default=3600,help='Maximum seconds waiting for each training-seed pair.')
    args=parser.parse_args()
    if args.timeout<0:parser.error('--timeout must be nonnegative')
    run(args.output,args.timeout)


if __name__=='__main__':main()
