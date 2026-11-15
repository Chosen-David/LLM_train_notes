'use strict';
const $=id=>document.getElementById(id);
const papers={
 zero:['ZeRO · 模型状态分片','https://arxiv.org/abs/1910.02054','把优化器状态、梯度、参数的冗余分别消除。对照本页的参数生命周期：存储分片和计算时权重可用性是两个层面。'],
 mega:['Megatron-LM · TP / PP / DP 组合','https://arxiv.org/abs/2104.04473','研究张量、流水线与数据并行的组合及其权衡。对照层内 TP 的部分和与跨层 PP 的激活传递。'],
 ulysses:['Ulysses · 改变 Attention 的切分维','https://arxiv.org/abs/2309.14509','通过 All-to-All 在序列切分与 heads 切分之间转换。它不是本图 AllGatherKV 的同名替换；每卡得到的张量布局不同。'],
 ring:['Ring Attention · 分块传递 K/V','https://arxiv.org/abs/2310.01889','沿设备环传递 K/V 块，把传输与分块 Attention 计算重叠，避免必须一次常驻全序列 K/V。对照本页的 CP 收集步骤。'],
 blocks:['MegaBlocks · 改变专家计算组织','https://arxiv.org/abs/2211.15841','以块稀疏计算适应动态专家负载，减少 padding 浪费并避免丢弃 token。它关注专家算子的组织，不等于动态搬迁专家布局。'],
 laer:['LAER-MoE · 解耦专家存储与计算布局','https://arxiv.org/abs/2602.11686','持久保存专家分片，按负载规划恢复计算副本。对照普通 EP 的固定专家归属与 expert FSDP：分片不自动带来负载自适应布局。']
};
const groups={
 dense:['普通 FSDP','dp_shard×CP：收集普通参数，归约其参数梯度；不是 Attention 激活的 CP 通信。'],
 expert:['专家 FSDP','edp_shard：同一专家集合的参数分片伙伴；各自 EP 组提供不同 token。'],
 tp:['TP','该轴可切 heads、词表或 FFN 特征；MoE routed 入口还会沿它切 token，含义随模块变化。'],
 cp:['CP','同一序列的不同 token 位置。此页面固定 AllGatherKV 后端；其它 CP 后端会采用不同通信。'],
 ep:['EP','不同专家编号之间的 token 分发 / 回收组；不能把不同专家参数当副本归约。'],
 rep:['DP replica','对应参数分片的副本，同步各自样本产生的梯度。'],
 pp:['PP','不同层所属的 stage：前向传激活，反向传激活梯度，不是对层权重做 All-Reduce。'],
 dps:['dp_shard 单轴','仅固定 CP、TP 后的 dp_shard 伙伴。默认普通 FSDP 还包含 CP，因此此组更小。'],
 none:['本地操作','当前子步骤不要求新增的跨 rank 激活通信；参数可能已由 FSDP 单元预先准备。']
};
function coords(rank){const p=Math.floor(rank/32),r=Math.floor(rank%32/16),local=rank%16;return {p,r,d:Math.floor(local/4),c:Math.floor(local%4/2),t:local%2,s:Math.floor(local/8),e:local%8};}
function groupMembers(rank,g){const a=coords(rank);return Array.from({length:64},(_,i)=>i).filter(i=>{const b=coords(i);switch(g){case 'dense':return a.p===b.p&&a.r===b.r&&a.t===b.t;case 'expert':return a.p===b.p&&a.r===b.r&&a.e===b.e;case 'tp':return a.p===b.p&&a.r===b.r&&a.d===b.d&&a.c===b.c;case 'cp':return a.p===b.p&&a.r===b.r&&a.d===b.d&&a.t===b.t;case 'ep':return a.p===b.p&&a.r===b.r&&a.s===b.s;case 'rep':return a.p===b.p&&i%16===rank%16;case 'pp':return i%32===rank%32;case 'dps':return a.p===b.p&&a.r===b.r&&a.c===b.c&&a.t===b.t;default:return i===rank;}});}
let current=0,step=0,rank=0;
function allowed(){const phase=$('phase').value;return PANELS.map((_,i)=>i).filter(i=>phase==='all'||phase==='fwd'&&i>=1&&i<=7||phase==='bwd'&&i>=8&&i<=12||phase==='state'&&(i===0||i===13));}
function options(){const ids=allowed();$('panel').replaceChildren(...ids.map(i=>new Option(PANELS[i].heading,i)));if(!ids.includes(current)){current=ids[0];step=0;}render();}
function renderGroup(){const g=$('group').value==='auto'?PANELS[current].zhsteps[step][1]:$('group').value,a=coords(rank),members=groupMembers(rank,g);$('ranktitle').textContent='GPU '+rank+' · '+groups[g][0];$('coordinates').textContent=`dense [pp=${a.p}, replica=${a.r}, dp_shard=${a.d}, cp=${a.c}, tp=${a.t}]\nexpert [pp=${a.p}, replica=${a.r}, edp_shard=${a.s}, ep=${a.e}]`;$('members').textContent='组成员：{'+members.join(', ')+'}';$('groupwhy').textContent=groups[g][1];
 $('grid').replaceChildren();for(let r=0;r<2;r++)for(let p=0;p<2;p++){const b=document.createElement('div');b.className='block';const h=document.createElement('h4');h.textContent=`PP stage ${p} · DP replica ${r}`;b.append(h);const cells=document.createElement('div');cells.className='ranks';for(let l=0;l<16;l++){const n=32*p+16*r+l,btn=document.createElement('button');btn.className='gpu'+(members.includes(n)?' member':'')+(rank===n?' focus':'');btn.textContent='GPU '+n;btn.setAttribute('aria-label','选择 GPU '+n);btn.setAttribute('aria-pressed',rank===n?'true':'false');btn.onclick=()=>{rank=n;renderGroup();saveHash();};cells.append(btn);}b.append(cells);$('grid').append(b);}}
function saveHash(){history.replaceState(null,'',`#module=${current}&step=${step}&gpu=${rank}`);}
function render(){const p=PANELS[current],ids=allowed();$('panel').value=current;$('counter').textContent=`模块 ${String(current).padStart(2,'0')} / 13`;$('heading').textContent=p.heading;$('explain').textContent=p.explain;$('why').textContent=p.why;$('code').textContent=p.code;$('state').textContent=p.zhsteps[step][2];$('steps').replaceChildren(...p.zhsteps.map((s,j)=>{const b=document.createElement('button');b.textContent=(j+1)+'. '+s[0];b.setAttribute('aria-pressed',j===step?'true':'false');b.onclick=()=>{step=j;render();};return b;}));
 $('diagram').src=`../assets/atlas/${String(current).padStart(2,'0')}.svg`;$('diagramlink').href=$('diagram').src;$('diagram').alt=p.heading+'：全局布局及模块计算流程';$('sources').replaceChildren(...p.src.map(([path,line])=>{const a=document.createElement('a');a.href=`https://github.com/pytorch/torchtitan/blob/631d544b1f6794ecfeca734a7bba861e03ddd5e3/torchtitan/${path}#L${line}`;a.textContent=path+':'+line;a.target='_blank';a.rel='noreferrer';return a;}));
 let keys=current===2||current===10?['ulysses','ring']:([3,4,5,8,9].includes(current)?['blocks','laer']:current===0||current>=12?['zero','laer']:['mega','zero']);$('relevance').textContent=current===2||current===10?'对照 CP 的 K/V 通信：可以改变数据重分布方式，也可以改变传输与计算的重叠方式。':[3,4,5,8,9].includes(current)?'对照专家路径：优化本地计算组织与改变专家副本布局，是不同层次的设计。':'对照参数存储和并行分工，区分显存节省、通信与调度的作用。';$('papers').replaceChildren(...keys.map(k=>{const [title,url,desc]=papers[k],d=document.createElement('div');d.className='paper';const a=document.createElement('a');a.href=url;a.target='_blank';a.rel='noreferrer';a.textContent=title;const p=document.createElement('p');p.textContent=desc;d.append(a,p);return d;}));$('prev').disabled=ids.indexOf(current)===0;$('next').disabled=ids.indexOf(current)===ids.length-1;renderGroup();saveHash();}
$('phase').onchange=options;$('panel').onchange=()=>{current=+$('panel').value;step=0;render();};$('group').onchange=renderGroup;for(const [id,delta] of [['prev',-1],['next',1]])$(id).onclick=()=>{const ids=allowed();current=ids[ids.indexOf(current)+delta];step=0;render();};
try{const h=new URLSearchParams(location.hash.slice(1));current=Math.min(13,Math.max(0,parseInt(h.get('module'))||0));step=Math.min(2,Math.max(0,parseInt(h.get('step'))||0));rank=Math.min(63,Math.max(0,parseInt(h.get('gpu'))||0));}catch{}options();
