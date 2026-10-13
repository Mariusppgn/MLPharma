// Headless unit checks, NOT a real browser or a visual accessibility audit.
import fs from 'node:fs/promises';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import {Blob} from 'node:buffer';

export async function runChecks(root) {
  const source = await fs.readFile(`${root}/src/deep_pharma/combination_profiles/assets/app.js`, 'utf8');
  const data = JSON.parse(await fs.readFile(`${root}/work/combination_profiles_smoke.json`, 'utf8'));
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) elements.set(id, {value:'',innerHTML:'',textContent:'',disabled:false,
      checked:false,hidden:false,dataset:{},setAttribute(){},replaceChildren(){this.innerHTML='';},
      classList:{add(){},remove(){}},addEventListener(){},removeAttribute(name){delete this[name];},
      showModal(){this.open=true;},close(){this.open=false;},clientWidth:1000});
    return elements.get(id);
  };
  const tabs = ['bars','heatmap','delta','scatter'].map(name => ({dataset:{view:name},disabled:false,setAttribute(){}}));
  const document = {getElementById:element,querySelectorAll:selector=>selector==='[data-view]'?tabs:[]};
  const context = vm.createContext({document,console,setTimeout,clearTimeout,Blob,URL});
  new vm.Script(source.replace(/init\(\);\s*$/, '')).runInContext(context);
  context.fixture = data;
  const evaluate = code => vm.runInContext(code,context);
  evaluate('catalog=fixture.catalog; result=fixture.context;');
  evaluate('render();');
  assert.equal(element('structures-section').hidden,false);
  assert.match(element('structure-title-a').textContent,new RegExp(data.context.selection.a));
  assert.match(element('structure-img-a').src,/^blob:/);
  evaluate('enlargeStructure("a");');
  assert.equal(element('structure-dialog').open,true);
  const originalWidth=element('structure-dialog-img').width;
  evaluate('$("structure-zoom").value="200";updateStructureZoom();');
  assert.equal(element('structure-dialog-img').width,originalWidth*2);
  evaluate('$("structure-dialog").close();');
  assert.equal((element('table').innerHTML.match(/<tr /g)||[]).length,50);
  assert.match(element('a-heading').textContent,/contexte/);
  assert.equal(tabs[2].disabled,true);
  assert.equal(tabs[3].disabled,true);
  assert.match(element('table').innerHTML,/Non calculable/);
  assert.match(element('pharma').innerHTML,/Mécanisme/);
  assert.equal(evaluate('resolve(catalog.default[0])'),data.catalog.default[0]);
  evaluate('autocomplete("CYP");');
  assert.ok(element('molecules-a').innerHTML.includes('<option'));
  evaluate('$("molecule-a").value=catalog.default[0];$("molecule-b").value=catalog.default[1];updatePartners();');
  const allowed=data.catalog.partners[data.catalog.default[0]];
  const suggestions=[...element('molecules-b').innerHTML.matchAll(/value="([^"]+)"/g)].map(m=>m[1]);
  assert.deepEqual(suggestions.sort(),[...allowed].sort());
  const incompatible=data.catalog.molecules.find(m=>!allowed.includes(m.id));
  if(incompatible){context.incompatible=incompatible.id;assert.throws(()=>evaluate('resolve(incompatible,partnerIds(catalog.default[0]))'));}
  const svgArtifacts = [];
  for(const name of ['bars','heatmap']) {
    evaluate(`view='${name}';renderChart();`);
    const html=element('chart').innerHTML;
    assert.equal((html.match(/data-effect=/g)||[]).length,50);
    assert.match(html,/viewBox/);
    svgArtifacts.push({name,svg:html});
  }
  evaluate('result=fixture.filtered;render();');
  const n=data.filtered.rows.length;
  assert.equal((element('table').innerHTML.match(/<tr /g)||[]).length,n);
  assert.equal((element('chart').innerHTML.match(/data-effect=/g)||[]).length,n);
  const csv=evaluate('csvString(result)');
  assert.equal(csv.split('\r\n').length,n+1);
  for(const row of data.filtered.rows) assert.ok(csv.includes(`"${row.effect}"`));
  assert.ok(csv.includes(data.catalog.audit.train_sha256));
  evaluate('result={...fixture.context,rows:[]};render();');
  assert.match(element('chart').innerHTML,/Aucun effet/);
  assert.equal(element('csv').disabled,true);
  evaluate('result=fixture.absent;render();');
  assert.ok(!element('table').innerHTML.includes('NaN'));
  // Explicit synthetic UNIT fixture only: exercises optional probability views, not production data.
  evaluate(`result=JSON.parse(JSON.stringify(fixture.context));result.selection.context=false;
    result.rows=result.rows.slice(0,2);result.rows.forEach((r,i)=>{
      r.reference=i===0?.2:.9;r.delta=i===0?.5:-.6;r.ab.value=i===0?.7:.3;
      r.ab.kind='probability';r.reason='SYNTHETIC UNIT FIXTURE';});`);
  for(const name of ['delta','scatter']) {
    evaluate(`view='${name}';renderChart();`);
    assert.equal((element('chart').innerHTML.match(/data-effect=/g)||[]).length,2);
    assert.equal(tabs[2].disabled,false);
    svgArtifacts.push({name:`unit-${name}`,svg:element('chart').innerHTML});
  }
  assert.match(evaluate('csvString(result)'),/"-60"/); // Numeric negative stays numeric.
  evaluate(`result.rows[0].label='=unsafe spreadsheet formula';`);
  assert.match(evaluate('csvString(result)'),/'=unsafe/);
  evaluate('result=fixture.context;selected=result.rows[0].effect;render();');
  assert.match(element('table').innerHTML,/<tr class="active"/);
  assert.match(element('chart').innerHTML,/circle cx="14"/);
  await fs.writeFile(`${root}/work/combination_profiles_svg_checks.json`,JSON.stringify(svgArtifacts));
  evaluate('clearResults();');
  assert.equal(element('structures-section').hidden,true);
  assert.equal(element('structure-img-a').src,undefined);
  return {passed:true,checks:26,svg_artifacts:svgArtifacts.length,
    note:'VM unit tests only; browser rendering not verified'};
}
