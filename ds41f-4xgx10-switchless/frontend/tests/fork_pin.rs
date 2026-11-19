use dsfront::session::{Plan,SessionTable};
use dsfront::tokenizer::{bytes_to_unicode,Tokenizer};
use serde_json::json;
use std::collections::HashMap;

fn tokenizer() -> Tokenizer {
    let vocab: HashMap<String,usize> = bytes_to_unicode().iter().enumerate()
        .map(|(i,c)| (c.to_string(),i)).collect();
    let config=json!({
        "added_tokens":[{"id":256,"content":"<U>","special":true,"normalized":false}],
        "model":{"type":"BPE","vocab":vocab,"merges":[]}
    });
    Tokenizer::from_json(&config.to_string()).unwrap()
}
fn text(id:char) -> String {
    format!("{}<U>{}",id.to_string().repeat(128),"tail".repeat(16))
}
fn complete(t:&mut SessionTable,tok:&Tokenizer,p:&Plan,prompt:&str) {
    t.terminal(p.session_id,p.base_session_id,false);
    t.finish(p.session_id,prompt,&p.ids,&[],tok);
}
fn put(t:&mut SessionTable,tok:&Tokenizer,prompt:&str) -> u64 {
    let (p,_) = t.plan(prompt,tok,1); complete(t,tok,&p,prompt); p.session_id
}

#[test]
fn production_cap_protects_an_oldest_fork_base() {
    let tok=tokenizer();
    let mut t=SessionTable::new(256,16<<20);
    let original=text('A');
    let base=put(&mut t,&tok,&original);
    let mut oldest_other=0;
    for i in 0..255 {
        let id=put(&mut t,&tok,&format!("{i:03}{}", "filler".repeat(16)));
        if i==0 {oldest_other=id;}
    }
    assert_eq!(t.len(),256);
    let (fork,release)=t.plan(&original.replace("tail","fork"),&tok,1);
    assert_eq!(fork.base_session_id,base);
    assert!(fork.reused>=128);
    assert!(!release.contains(&base),"selected fork base released before submit");
    assert_eq!(release,vec![oldest_other]);
    assert_eq!(t.len(),256);
}

#[test]
fn last_dependent_controls_reuse_even_when_client_finishes_early() {
    let tok=tokenizer(); let mut t=SessionTable::new(12,16<<20);
    let original=text('A'); let base=put(&mut t,&tok,&original);
    let b=original.replace("tail","branch"); let c=original.replace("tail","fork");
    let (p,_) = t.plan(&b,&tok,1);
    let (q,_) = t.plan(&c,&tok,1);
    assert_eq!((p.base_session_id,q.base_session_id),(base,base));
    t.finish(p.session_id,&b,&p.ids,&[],&tok);
    t.terminal(q.session_id,q.base_session_id,true);
    let (r,release)=t.plan(&original,&tok,1);
    assert_eq!(r.base_session_id,base);
    assert!(!release.contains(&base));
    t.terminal(r.session_id,r.base_session_id,true);
    t.terminal(p.session_id,p.base_session_id,false);
    let (next,_) = t.plan(&original,&tok,1);
    assert_eq!((next.session_id,next.base_session_id),(base,0));
}

#[test]
fn failed_in_place_parent_releases_once_after_last_queued_fork() {
    let tok=tokenizer(); let mut t=SessionTable::new(20,16<<20);
    let orig=text('A'); let base=put(&mut t,&tok,&orig);
    let (parent,_)=t.plan(&orig,&tok,1);
    assert_eq!(parent.session_id,base);
    let (a,_)=t.plan(&orig.replace("tail","fork"),&tok,1);
    let (b,_)=t.plan(&orig.replace("tail","branch"),&tok,1);
    assert_eq!((a.base_session_id,b.base_session_id),(base,base));
    assert!(!t.terminal(base,0,true).contains(&base));
    let (next,_) = t.plan(&orig,&tok,1);
    assert_ne!(next.session_id,base); assert_ne!(next.base_session_id,base);
    assert!(!t.terminal(a.session_id,base,true).contains(&base));
    let released=t.terminal(b.session_id,base,true);
    assert_eq!(released.iter().filter(|&&id|id==base).count(),1);
}

#[test]
fn byte_pressure_cannot_evict_reserved_source() {
    let tok=tokenizer(); let orig=text('A'); let other=text('B');
    let cap=tok.encode(&orig).len()+tok.encode(&other).len();
    let mut t=SessionTable::new(20,cap);
    let base=put(&mut t,&tok,&orig);
    let old=put(&mut t,&tok,&other);
    let fork_text=orig.replace("tail","split");
    let (fork,_)=t.plan(&fork_text,&tok,1);
    let released=t.finish(fork.session_id,&fork_text,&fork.ids,&[],&tok);
    assert_eq!(released,vec![old]);
    assert!(!released.contains(&base));
    let release=t.terminal(fork.session_id,base,false);
    assert!(release.contains(&base));
    assert_eq!(t.len(),1);
}
