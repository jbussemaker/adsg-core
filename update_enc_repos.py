"""
MIT License

Copyright: (c) 2024, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This file updates encoder code from external repositories:
- adsg_core.optimization.assign_enc from https://github.com/jbussemaker/AssignmentEncoding
- adsg_core.optimization.sel_choice_enc from https://github.com/jbussemaker/SelectionChoiceEncoding
Only use if needed!!
"""
import os
import glob
import shutil
import tempfile
import subprocess

cwd = os.path.dirname(__file__)


def update_from_external(target_module, target_tests_module, repo_url, base_module):

    def _copy_tree(src_dir, tgt_dir):
        shutil.rmtree(tgt_dir)
        shutil.copytree(src_dir, tgt_dir)

        # Remove .pytest_cache folders
        for cache_folder in glob.glob(f'{tgt_dir}/**/.pytest_cache', recursive=True):
            shutil.rmtree(cache_folder)

    def _provision(tgt_dir):
        for file in glob.glob(f'{tgt_dir}/**/*.py', recursive=True):
            with open(file, 'r') as fp:
                contents = fp.read()

            # Modify module import paths
            if f'{base_module}.' in contents:
                contents = contents.replace(f' {base_module}.', f' {target_module}.')
                contents = contents.replace('from tests.', f'from {target_tests_module}.')
                with open(file, 'w') as fp:
                    fp.write(contents)

    with tempfile.TemporaryDirectory() as tmp_folder:
        # Clone git repo
        print(f'Cloning {repo_url}')
        subprocess.run(['git', 'clone', repo_url, '.'], cwd=tmp_folder)

        # Move code
        print('Moving code and tests')
        adsg_tgt_dir = f'{cwd}/{target_module.replace(".", "/")}'
        repo_src_dir = f'{tmp_folder}/{base_module}'
        _copy_tree(repo_src_dir, adsg_tgt_dir)
        _provision(adsg_tgt_dir)

        # Move tests
        adsg_test_dir = f'{cwd}/{target_tests_module.replace(".", "/")}'
        repo_test_dir = f'{tmp_folder}/tests'
        _copy_tree(repo_test_dir, adsg_test_dir)
        _provision(adsg_test_dir)


def change_assign_enc_names(assign_enc_module, name):
    print('Updating assign_enc names')
    adsg_assign_enc_dir = f'{cwd}/{assign_enc_module.replace(".", "/")}'

    filename = f'{adsg_assign_enc_dir}/cache.py'
    with open(filename, 'r') as fp:
        contents = fp.read()
    contents = contents.replace('AssignmentEncoder', name)
    with open(filename, 'w') as fp:
        fp.write(contents)

    filename = f'{adsg_assign_enc_dir}/selector.py'
    with open(filename, 'r') as fp:
        contents = fp.read()
    contents = contents.replace('assign_enc.selector', f'{assign_enc_module.split(".")[0]}.selector')
    contents = contents.replace('log.info(', 'log.debug(')
    with open(filename, 'w') as fp:
        fp.write(contents)


if __name__ == '__main__':
    update_from_external(
        target_module='adsg_core.optimization.assign_enc',
        target_tests_module='adsg_core.tests.assign_enc',
        repo_url='https://github.com/jbussemaker/AssignmentEncoding',
        base_module='assign_enc',
    )
    change_assign_enc_names(
        assign_enc_module='adsg_core.optimization.assign_enc',
        name='ADSG'
    )

    update_from_external(
        target_module='adsg_core.optimization.sel_choice_enc',
        target_tests_module='adsg_core.tests.sel_choice_enc',
        repo_url='https://github.com/jbussemaker/SelectionChoiceEncoding',
        base_module='sel_choice_enc',
    )
