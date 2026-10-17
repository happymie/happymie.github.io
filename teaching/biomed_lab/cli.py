"""CLI entrypoint. External downloads happen only in fetch-upstream."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
from . import data,registry


def main(argv=None):
    parser=argparse.ArgumentParser(description='Biomedical AI Atlas: inspect, learn, run and verify')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('list');p.add_argument('--runnable',action='store_true')
    p=sub.add_parser('info');p.add_argument('model')
    p=sub.add_parser('data');p.add_argument('model',help='Model number, exact name, or all');p.add_argument('--inspect',action='store_true')
    p=sub.add_parser('run');p.add_argument('model');p.add_argument('--steps',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=2);p.add_argument('--seed',type=int,default=2026);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--device',default='cpu');p.add_argument('--max-seconds',type=float,default=120);p.add_argument('--max-rss-mb',type=float,default=3072)
    p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--set',action='append',default=[],metavar='KEY=JSON')
    p.add_argument('--out-dir',type=Path);p.add_argument('--save-checkpoint',action='store_true')
    p=sub.add_parser('fetch-upstream');p.add_argument('model');p.add_argument('--destination',type=Path,default=registry.ROOT/'upstream')
    p=sub.add_parser('verify');p.add_argument('model',nargs='?',default='all')
    p.add_argument('--steps',type=int,default=2);p.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    try:
        if args.command=='list':
            rows=registry.entries(); available=registry.specs()
            for m in rows:
                spec=available.get(m['num'])
                if args.runnable and not spec:continue
                print(f"{m['num']:03d}  {m['title']:<36} {(spec or {}).get('implementation','source_route')}")
            print('upstream = adapted/open library code; paper_port = independent implementation; mechanism_demo = partial mechanism; source_route = no CPU adapter')
        elif args.command=='info':
            m=registry.resolve(args.model)
            obj={**m,'source':registry.source(m['num']),'adapter':registry.specs().get(m['num']),
                 'database':data.inspect_database(data.create_database(m))}
            print(json.dumps(obj,ensure_ascii=False,indent=2))
        elif args.command=='data':
            rows=registry.entries() if args.model=='all' else [registry.resolve(args.model)]
            for m in rows:
                p=data.create_database(m)
                print(json.dumps(data.inspect_database(p),ensure_ascii=False) if args.inspect else str(p))
        elif args.command=='run':
            from .runner import run
            overrides={}
            for item in args.set:
                key,value=item.split('=',1);overrides[key]=json.loads(value)
            result,folder=run(args.model,args.steps,args.batch_size,args.seed,args.threads,args.device,overrides,args.out_dir,
                              args.max_seconds,args.max_rss_mb,args.lr,args.save_checkpoint)
            print(json.dumps({k:v for k,v in result.items() if k not in {'losses','limitations'}},ensure_ascii=False,indent=2))
            print(f'Results: {folder}')
            if result['status']!='completed': return sys.exit(1)
        elif args.command=='verify':
            from .verification import verify
            results=verify(args.model,steps=args.steps)
            if args.output:
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
                print(f'Execution records: {args.output}')
            else: print(json.dumps(results,ensure_ascii=False,indent=2))
            if any(x['status'] in {'failed','budget_exceeded'} for x in results):return sys.exit(1)
        elif args.command=='fetch-upstream':
            s=registry.source(args.model);repo=s.get('repo_url');sha=s.get('commit')
            if not repo or not repo.startswith('https://github.com/') or not sha:
                raise ValueError(f"No pinned Git repository for this item. Review {s.get('source_url') or s['paper_url']} manually; gated weights are never fetched automatically.")
            dest=args.destination.resolve()/f"{s['num']:03d}-{repo.rsplit('/',1)[-1]}"
            if dest.exists():raise ValueError(f'Refusing to replace existing checkout: {dest}')
            print('Upstream license:',s.get('license_spdx'),'—',s.get('license_url'))
            print('Fetching source only. No installation, checkpoint download, or upstream script execution.')
            dest.parent.mkdir(parents=True,exist_ok=True)
            subprocess.run(['git','init',str(dest)],check=True)
            subprocess.run(['git','-C',str(dest),'remote','add','origin',repo+'.git'],check=True)
            subprocess.run(['git','-C',str(dest),'fetch','--depth','1','origin',sha],check=True)
            subprocess.run(['git','-C',str(dest),'checkout','--detach','FETCH_HEAD'],check=True)
            actual=subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True).strip()
            if actual!=sha:raise ValueError('Fetched revision does not match the source registry')
            print(f'Pinned source: {dest}\nReview the upstream README and license before the original workflow.')
    except (ValueError,KeyError,FileNotFoundError,subprocess.CalledProcessError) as exc:
        parser.exit(2,f'Error: {exc}\n')


if __name__=='__main__':main()
