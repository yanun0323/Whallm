import tempfile
from pathlib import Path
import unittest

from Scripts.analyze_qwen_metal import merge_intervals, summarize, table_rows


class QwenMetalTests(unittest.TestCase):
    def fixture(self):
        timeline=[dict(kind='metadata',clock='mach_absolute_time',mode='sync',pid=42),
            dict(kind='decoder',request=0,phase='decode',layer=0,begin=0,end=10,failed=False),
            dict(kind='decoder',request=0,phase='decode',layer=1,begin=10,end=20,failed=False),
            dict(kind='phase',request=0,phase='decode',layer=-1,begin=0,end=20,failed=False)]
        rows=[dict(start=str(a),duration=str(b-a),cmdbuffer_id=str(i)) for i,(a,b) in enumerate(((1,6),(2,4),(5,12),(18,20)),1)]
        for row in rows:row.update(process='python (42)',state='Active');row['cmdbuffer-id']=row.pop('cmdbuffer_id')
        rows.append(dict(start='0',duration='20',process='other (7)',state='Active',**{'cmdbuffer-id':'9'}))
        return timeline,rows,{'mabs-epoch':'0','timebase-info':'1/1'}

    def test_union_excludes_other_process_and_does_not_double_count_nested_intervals(self):
        timeline,rows,info=self.fixture()
        result=summarize(timeline,rows,info,pid=42)
        layers={r['layer']:r for r in result if r['kind']=='decoder'}
        self.assertAlmostEqual(layers[0]['gpu_active_seconds'],9e-9)
        self.assertAlmostEqual(layers[1]['gpu_active_seconds'],4e-9)
        phase=next(r for r in result if r['kind']=='phase')
        self.assertAlmostEqual(phase['gpu_active_seconds'],13e-9)
        self.assertEqual(phase['command_buffer_ids'],4)
        self.assertAlmostEqual(phase['gpu_active_percent'],65)
        self.assertEqual(merge_intervals([]),[])
        with self.assertRaises(ValueError):merge_intervals([(5,1)])

    def test_missing_gpu_host_failed_wrong_pid_and_overlapping_scopes_are_rejected(self):
        for change in ('host','pid','failed','overlap','missing','tail'):
            timeline,rows,info=self.fixture()
            if change=='host':timeline[0]['mode']='host'
            if change=='pid':timeline[0]['pid']=9
            if change=='failed':timeline[1]['failed']=True
            if change=='overlap':timeline[2]['begin']=9
            if change=='missing':rows=[]
            if change=='tail':timeline[-1]['end']=1_000_000_000
            with self.subTest(change=change),self.assertRaises(ValueError):
                summarize(timeline,rows,info,pid=42)

    def test_timebase_and_phase_containment(self):
        timeline,rows,info=self.fixture()
        info['mabs-epoch']='100';info['timebase-info']='125/3'
        for row in timeline[1:]:row['begin']+=100;row['end']+=100
        for row in rows:
            row['start']=str(round(int(row['start'])*125/3))
            row['duration']=str(round(int(row['duration'])*125/3))
        result=summarize(timeline,rows,info,pid=42)
        self.assertAlmostEqual(next(r for r in result if r['kind']=='phase')['wall_seconds'],20*125/3/1e9)
        timeline[-1]['begin']=101
        with self.assertRaisesRegex(ValueError,'outside'):
            summarize(timeline,rows,info,pid=42)

    def test_xml_backward_references(self):
        xml='''<trace-query-result><node><schema><col><mnemonic>start</mnemonic></col><col><mnemonic>process</mnemonic></col></schema>
        <row><start-time id="1">5</start-time><process id="2" fmt="python (42)"><pid id="3">42</pid></process></row>
        <row><start-time ref="1"/><process ref="2"/></row></node></trace-query-result>'''
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'table.xml';path.write_text(xml)
            self.assertEqual(list(table_rows(path)),[dict(start='5',process='python (42)')]*2)


if __name__=='__main__':unittest.main()
