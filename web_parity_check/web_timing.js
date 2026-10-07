// Run in the browser console on https://drfaizb-thermal-paddy-classifier.static.hf.space/index.html after "Model ready".
// Uses the page's own `model` and `preprocess`. Results are in web_timing_M3.json.
// 1) Per-image time (same region the app times)
const c=document.createElement('canvas');c.width=320;c.height=240;const x=c.getContext('2d');const d=x.createImageData(320,240);
for(let i=0;i<d.data.length;i++)d.data[i]=(i%4===3)?255:Math.random()*255;x.putImageData(d,0,0);
for(let i=0;i<5;i++) tf.tidy(()=>model.predict(preprocess(c))).dataSync();       // warm-up
const ts=[];for(let i=0;i<100;i++){const t0=performance.now();tf.tidy(()=>model.predict(preprocess(c))).dataSync();ts.push(performance.now()-t0);}
ts.sort((a,b)=>a-b); console.log('median ms', ts[49]);
// 2) Model download with cache disabled
const files=['model.json',...Array.from({length:12},(_,i)=>`group1-shard${i+1}of12.bin`)];
let t0=performance.now(); await Promise.all(files.map(async f=>(await fetch(f+'?nc='+Date.now(),{cache:'no-store'})).arrayBuffer()));
console.log('download s', (performance.now()-t0)/1000);
// 3) Page-ready time in a fresh context
for(let r=0;r<3;r++){const f=document.createElement('iframe');f.style.cssText='position:absolute;left:-2000px;width:900px;height:700px';
  const s0=performance.now();f.src='index.html?r='+r+'_'+Date.now();document.body.appendChild(f);
  while(true){await new Promise(z=>setTimeout(z,20));const s=f.contentDocument&&f.contentDocument.getElementById('status');if(s&&/Model ready/.test(s.textContent))break;}
  console.log('ready ms', performance.now()-s0); f.remove(); await new Promise(z=>setTimeout(z,1500));}
