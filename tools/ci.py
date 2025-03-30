
import argparse
import os

def clean():
    print('Cleaning...')
    os.system('./clean.sh')
    print('Cleaned.')

def init():
    print('Initializing...')
    os.system('./init.sh')
    print('Initialized.')

def build():
    print('Building...')
    os.system('./build.sh')
    print('Built.')

def test():
    print('Testing...')
    os.system('./test.sh')
    print('Tested.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stages', nargs='+', type=str)
    args = parser.parse_args()
    print(f'Stages: {args.stages}')
    for stage in args.stages:
        if stage == 'clean':
            clean()
        elif stage == 'init':
            init()
        elif stage == 'build':
            build()
        elif stage == 'test':
            test()
        else:
            print(f'Unknown stage: {stage}')
            exit(1)