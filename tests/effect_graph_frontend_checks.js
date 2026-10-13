// Pure-function checks with a minimal fake document; not a browser/render test.
function frontendChecks(vm, source, data) {
  const elements = new Map();
  const defaults = {similarity:'0', weight:'0', topK:'100', method:'neighborhood',
    minFrequency:'0', minPairs:'0', prrMin:'0'};
  const element = id => {
    if (!elements.has(id)) elements.set(id, {value:defaults[id]??'', innerHTML:'',
      textContent:'', selectedOptions:[], checked:id==='fullSelected', style:{},
      addEventListener(){}, querySelectorAll(){return []}, insertAdjacentHTML(){}});
    return elements.get(id);
  };
  const sandbox = {DATA:data, INITIAL_VIEW:'graph', document:{getElementById:element,
    querySelectorAll(){return []}}, console, setTimeout, clearTimeout};
  vm.createContext(sandbox);
  const setup = source.slice(0,source.lastIndexOf('setupDiscoveryUI();evaluationView();'));
  new vm.Script(setup).runInContext(sandbox);
  const assert = (expression,label) => {
    if (!vm.runInContext(expression,sandbox)) throw Error(label);
  };
  const pair = data.pairs[0];
  element('drugA').value=pair.a; element('drugB').value=pair.b;
  assert('predicate(DATA.pairs[0])','Exact known pair must be retained');
  element('drugA').value=pair.b; element('drugB').value=pair.a;
  assert('predicate(DATA.pairs[0])','A/B swapping must preserve pair selection');
  element('drugA').value='not-a-known-molecule';
  assert('!predicate(DATA.pairs[0])','Unknown molecule must not invent matches');
  element('drugA').value=''; element('drugB').value='';
  assert('DATA.pairs.every(predicate)','Default filters must preserve every pair');
  assert('populationStats(DATA.pairs).n===DATA.profile.pairs','Population counts must match');
  assert('csv([{name:"=SUM(A1)"}]).includes("\'=SUM(A1)")','CSV formula escaping');
  assert('esc("<script>").includes("&lt;script&gt;")','Text must be escaped');
  assert('wrapNodeLabel("Insuffisance rénale aiguë").replaceAll("\\n"," ")==="Insuffisance rénale aiguë"','Labels must retain their entire text');
  assert('nodeGeometry({id:24,filtered_frequency:.2}).nodeHeight>52','Long labels need a taller node');
  sandbox.history={replaceState(){}}; sandbox.location={hash:''};
  let zoom=1;
  sandbox.cytoscape=()=>({on(){},resize(){},fit(){},add(){},pan(){},
    zoom(value){if(value!==undefined)zoom=value;return zoom;},
    elements(){return {remove(){},components(){return []}}},
    nodes(){return {forEach(){}}}, layout(){return {run(){}}}});
  new vm.Script('evaluationView();diagnosticCharts();update();show(INITIAL_VIEW);').runInContext(sandbox);
  if(!element('summary').innerHTML.includes('Paires filtrées')) throw Error('Missing summary');
  element('drugA').value='unknown';
  vm.runInContext('update()',sandbox);
  assert('visibleNodes.length===0&&visibleEdges.length===0','An empty population must not display a global graph');
  return {checks:11,status:'passed',scope:'Pure functions and mocked initialization; no visual/browser validation'};
}
