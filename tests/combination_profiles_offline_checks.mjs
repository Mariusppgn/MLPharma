// Executes the actual embedded scripts/data in a VM, not a substitute for browser visual QA.
import fs from 'node:fs/promises';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import {gunzipSync} from 'node:zlib';
import {DecompressionStream} from 'node:stream/web';
import {Blob,Buffer} from 'node:buffer';

export async function runChecks(root) {
  const html=await fs.readFile(`${root}/outputs/combination_profiles/DeepPharma_Combinaisons.html`,'utf8');
  const encoded=html.match(/id="deep-pharma-snapshot"[^>]*>(.*?)<\/script>/s)[1];
  const data=JSON.parse(gunzipSync(Buffer.from(encoded,'base64')).toString('utf8'));
  const scripts=[...html.matchAll(/<script>(.*?)<\/script>/gs)].map(m=>m[1]).join('\n');
  const elements=new Map();
  const element=id=>{
    if(!elements.has(id))elements.set(id,{textContent:id==='deep-pharma-snapshot'?encoded:'',
      value:id==='source'?'binary':id==='threshold'?'0.5':id==='margin'?'0.05':id==='min-support'?'0':'',
      checked:id==='context',disabled:false,innerHTML:'',hidden:false,setAttribute(){},
      replaceChildren(){this.innerHTML='';},addEventListener(type,fn){this[type]=fn;},
      insertAdjacentHTML(where,text){this.innerHTML+=text;},classList:{add(){},remove(){}},
      removeAttribute(name){delete this[name];},showModal(){this.open=true;},close(){this.open=false;},clientWidth:1000});
    return elements.get(id);
  };
  const document={getElementById:element,querySelectorAll:()=>[]};
  const context=vm.createContext({document,console,setTimeout,clearTimeout,Uint8Array,Blob,Response,URL,
    DecompressionStream,atob:text=>Buffer.from(text,'base64').toString('binary')});
  new vm.Script(scripts.replace(/init\(\);\s*$/,'')).runInContext(context);
  // Exercise the actual inline gzip loading path, with no fetch supplied at all.
  const catalog=await vm.runInContext('DeepPharmaOffline.get("/api/catalog")',context);
  assert.equal(catalog.molecules.length,958);
  assert.deepEqual(JSON.parse(JSON.stringify(catalog.partners)),data.catalog.partners);
  await vm.runInContext('init()',context);
  assert.equal(element('status').classList.error,undefined);
  assert.match(element('status').textContent,/Train_/);
  assert.match(element('table').innerHTML,/Céphalée/);
  assert.match(element('structure-img-a').src,/^blob:/);
  assert.match(element('structure-title-a').textContent,new RegExp(catalog.default[0]));
  const previousA=element('structure-img-a').src;
  vm.runInContext('enlargeStructure("b")',context);
  assert.equal(element('structure-dialog').open,true);
  assert.match(element('structure-dialog-title').textContent,new RegExp(catalog.default[1]));
  vm.runInContext('$("structure-dialog").close()',context);
  assert.ok((element('molecules-b').innerHTML.match(/<option /g)||[]).length>0);
  const cases=JSON.parse(await fs.readFile(`${root}/work/combination_profiles_offline_parity.json`,'utf8'));
  const provider=await vm.runInContext('DeepPharmaOffline.ready',context);
  for(const {request,expected} of cases){
    const response=JSON.parse(JSON.stringify(provider.compare(request)));
    for(const [key,value] of Object.entries(expected))assert.deepEqual(response[key],value,`${request.source} ${request.pair_id||request.a} ${key}`);
  }
  const a=catalog.default[0],b=catalog.default[1];
  const bad=catalog.molecules.find(m=>!catalog.partners[a].includes(m.id));
  if(bad)assert.throws(()=>provider.compare({a,b:bad.id}),/partenaire/);
  assert.throws(()=>provider.compare({a,b,pair_id:'invalid'}),/Pair_ID/);
  for(const m of catalog.molecules){
    for(const partner of catalog.partners[m.id]){
      assert.ok(data.pairs[[m.id,partner].sort().join('\x1f')]);
      assert.ok(catalog.partners[partner].includes(m.id));
    }
  }
  // An A change cannot retain an invalid B, and old results/export buttons are cleared.
  context.nextA=catalog.molecules.find(m=>m.id!==a && catalog.partners[m.id].length).id;
  vm.runInContext('$("molecule-a").value=nextA;clearResults();updatePartners();',context);
  assert.ok(catalog.partners[context.nextA].includes(element('molecule-b').value));
  assert.equal(element('csv').disabled,true);
  assert.equal(element('table').innerHTML,'');
  assert.equal(element('structures-section').hidden,true);
  assert.equal(element('structure-img-a').src,undefined);
  await vm.runInContext('compare()',context);
  assert.equal(element('structures-section').hidden,false);
  assert.notEqual(element('structure-img-a').src,previousA);
  assert.match(element('structure-title-a').textContent,new RegExp(context.nextA));
  assert.ok(!/<script\s+src=|<link[^>]+href=/i.test(html));
  assert.ok(html.includes("connect-src 'none'"));
  return {passed:true,python_js_parity_cases:cases.length,canonical_pairs_checked:Object.keys(data.pairs).length,
    gzip_loading:true,no_fetch_available:true,html_bytes:Buffer.byteLength(html),
    note:'Actual bundled JS and snapshot tested in VM; browser rendering not verified'};
}
