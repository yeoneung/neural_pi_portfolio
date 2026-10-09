"""Single entry point for portfolio-control experiments and numerical reports."""
import argparse
import subprocess
import sys
from followup_io import ROOT, BOUNDARY_DEFAULT, DERIVATIVE_DEFAULT, project_path


def python_script(name, *args):
    subprocess.run([sys.executable,str(ROOT/'scripts'/name),*map(str,args)],cwd=ROOT,check=True)


def report(boundary, derivatives, tables):
    python_script('report_continuous_followup.py','--boundary-results',boundary,
                  '--derivative-results',derivatives,'--tables-dir',tables)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('report',help='Verify saved inputs and recompute all follow-up comparisons and tables.')
    p.add_argument('--boundary-results',type=project_path,default=BOUNDARY_DEFAULT)
    p.add_argument('--derivative-results',type=project_path,default=DERIVATIVE_DEFAULT)
    p.add_argument('--tables-dir',type=project_path,default=ROOT/'reports/tables')
    p=commands.add_parser('rerun',help='Recompute both studies in a separate directory, using the recorded prior checkpoints/references.')
    p.add_argument('--output',type=project_path,required=True)
    p.add_argument('--resume',action='store_true',help='Skip completed analyses/cases; incomplete training cases are not overwritten.')
    p=commands.add_parser('controls',help='Run or verify the matched CPU, quotient, and damping comparisons.')
    p.add_argument('--run',action='store_true',help='Generate missing controlled experiments before reporting.')
    p=commands.add_parser('matched-derivatives',help='Verify or compute shared references for the matched CPU designs.')
    p.add_argument('--run',action='store_true',help='Solve missing references and extract derivative arrays before reporting.')
    p.add_argument('--output',type=project_path,default=ROOT/'results/npi_matched_derivatives_20260929')
    p.add_argument('--report-dir',type=project_path,default=ROOT/'reports',help='Destination for the diagnostic figure and tables.')
    args=parser.parse_args()
    if args.command=='report':
        report(args.boundary_results,args.derivative_results,args.tables_dir)
    elif args.command=='rerun':
        args.output.mkdir(parents=True,exist_ok=args.resume)
        boundary=args.output/'boundary';derivatives=args.output/'derivatives'
        resume=['--resume'] if args.resume else []
        python_script('followup_derivative_curves.py','--output',derivatives,*resume)
        for phase in ['frozen','iteration','simulation']:
            python_script('followup_continuous_boundary.py','--phase',phase,'--output',boundary)
        report(boundary,derivatives,args.output/'tables')
    elif args.command=='controls':
        if args.run:python_script('evaluation_controls.py','all')
        python_script('report_evaluation_controls.py')
    elif args.command=='matched-derivatives':
        if args.run:python_script('matched_derivative_bridge.py','--output',args.output)
        python_script('report_matched_derivatives.py','--output',args.output,'--report-dir',args.report_dir)



if __name__=='__main__':main()
