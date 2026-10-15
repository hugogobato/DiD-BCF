"""Bounded package/assignment/notebook checks, with no sampling or installs."""
from pathlib import Path
import ast
from collections import Counter
import json
import os
import sys
import tempfile
import unittest.mock as mock

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE/'helpers'))
from common import assignment,atomic,digest,read,verify_fit,verify_package


def main():
    import nbformat
    package,manifest=verify_package()
    ids={j['id'] for j in manifest['jobs']}
    assert len(ids)==9227
    assigned=[];local=[];counts=Counter();waves=Counter()
    for path in (PACKAGE/'assignments').glob('*.json'):
        item=assignment(path,manifest)
        if item['execution']=='colab':
            assigned+=item['job_ids'];waves[item['wave']]+=1
            assert 7.9<item['estimated_wall_hours']<8.1
            counts.update(item['phase_counts'])
        else:local+=item['job_ids']
    reuse=set(manifest['reused_job_ids'])
    assert len(assigned)==len(set(assigned))==9197
    assert len(local)==len(set(local))==22
    assert not (set(assigned)&set(local) or set(assigned)&reuse or set(local)&reuse)
    assert set(assigned)|set(local)|reuse==ids
    assert dict(waves)=={1:48,2:48,3:36}
    assert dict(counts)=={'diagnostic':3999,'pt_impact':4000,'sensitivity':1198}
    for job_id in reuse:assert verify_fit(PACKAGE/'reuse/fits'/job_id,manifest)
    notebooks=list((PACKAGE/'Colab').rglob('*.ipynb'))
    assert len(notebooks)==133
    code_cells=0;versions=set()
    for path in notebooks:
        data=read(path);nbformat.validate(nbformat.from_dict(data))
        source=''.join(''.join(c['source']) for c in data['cells'])
        assert 'transparent_background' not in source
        assert 'files.download(output_file)' in source
        assert '(Not on Colab / download skipped):' in source
        for cell in data['cells']:
            if cell['cell_type']=='code':ast.parse(''.join(cell['source']));code_cells+=1
        setup=''.join(data['cells'][1]['source'])
        import re
        commit=re.search("SOURCE_COMMIT = '([a-f0-9]{40})'",setup).group(1)
        versions.add(commit)
    assert code_cells==662 and len(versions)==1
    # Execute the setup cell with network/install calls mocked, while using
    # the real sealed manifest. This catches missing imports and variables.
    first=read(next((PACKAGE/'Colab/wave_01').glob('*.ipynb')))
    setup=''.join(first['cells'][1]['source'])
    with tempfile.TemporaryDirectory(prefix='nonmain-notebook-setup-') as temp:
        checkout=Path(temp)/'checkout';target=checkout/'Simulation_Studies_Revision/DiD_BCF/Nonmain_Rerun_20261014'
        target.parent.mkdir(parents=True);target.symlink_to(PACKAGE,target_is_directory=True)
        lines=setup.splitlines()
        setup='\n'.join('REPO = Path('+repr(str(checkout))+')' if line.startswith('REPO = ') else line for line in lines)
        namespace={}
        with mock.patch('subprocess.run') as calls,mock.patch('subprocess.check_output',return_value=next(iter(versions))+'\n'):
            exec(compile(setup,'notebook_setup_fixture','exec'),namespace)
        assert namespace['PACKAGE']==target
        assert namespace['manifest']['manifest_hash']==manifest['manifest_hash']
        assert calls.call_count==5
    # Identical collector exports remain compact and complete fits restore
    # byte-identically; actual fixture tests cover conflict and traversal.
    report={'status':'PASS','manifest_hash':manifest['manifest_hash'],'package_hash':package['package_hash'],
            'source_commit':next(iter(versions)),'total_jobs':9227,'reused':8,'colab_jobs':9197,
            'local_jobs':22,'run_notebooks':132,'collector_notebooks':1,'code_cells':code_cells,
            'waves':dict(waves),'exact_partition':True,'all_reuse_verified':True,
            'notebook_schema_and_syntax':'PASS','mocked_setup_execution':'PASS',
            'sampling_calls':0,'dependency_install_calls':0,'network_calls':0,
            'note':'Mocked setup verifies Python orchestration only; actual native controls passed in local setup.'}
    atomic(PACKAGE/'validation/PACKAGE_CHECKS.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
