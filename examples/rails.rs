//! The Rails face: every memory of the sdm family declared with a builder, written by name and read with a
//! chain of keywords. Each block below is the Rust form of a SETTLE program in `settle-rs/examples/` (named
//! above it), and prints the very lines that program prints.
//!
//! Run: `cargo run --release --example rails`. The recorded output is `examples/rails.out`; every line under the
//! first two headers was compared with SETTLE's own output for those programs and is byte-identical
//! (settle-rs/tests/kanerva_rails.rs runs that comparison live).

use kanerva::rails::{Answer, ContentTrack, Error, Mode, Refusal, Sdm, SdmScale, SoftSdm, Via, Wake, RUN_SEED};
use kanerva::Rng;
use std::fmt::Write;

struct Out<'a>(&'a mut String);

impl Out<'_> {
    fn line(&mut self, l: &str) {
        writeln!(self.0, "{}", l).unwrap();
    }
    fn show(&mut self, r: Result<Answer, Error>) {
        match r {
            Ok(a) => writeln!(self.0, "{}", a).unwrap(),
            Err(e) => writeln!(self.0, "refused: {}", e).unwrap(),
        }
    }
}

/// The whole output, so tests/examples.rs can compare it with the record.
pub fn run() -> String {
    let mut out = String::new();
    programs(&mut out).unwrap_or_else(|e| writeln!(out, "refused: {}", e).unwrap());
    out
}

fn programs(out: &mut String) -> Result<(), Error> {
    let mut o = Out(out);
    // settle-rs/examples/sdm.settle
    o.line("== sdm");
    let mut s = Sdm::build().name("s").word_size(256).hard_locations(2000);
    s.write("cat")?;
    s.write("dog")?;
    s.write("owl")?;
    s.write_text("note", "meet at nine")?;
    o.show(s.read("cat").address_noise(0.2).seed(1).answer());
    o.show(s.read("cat").address_noise(0.2).via(Via::Pulls).seed(1).answer());
    o.show(s.read("note").address_noise(0.15).seed(2).answer());
    o.show(s.read("owl").address_noise(0.4).seed(3).answer());
    o.show(s.read_from_noise().seed(4).answer());
    o.show(s.read("zebra").address_noise(0.0).seed(5).answer());

    // settle-rs/examples/softsdm.settle: two memories read in one run block share one stream
    o.line("== softsdm");
    let mut hard = SoftSdm::build().name("hard").word_size(256).hard_locations(2000).activation_probability(0.05).softness(0.0);
    let mut soft = SoftSdm::build().name("soft").word_size(256).hard_locations(2000).activation_probability(0.05).softness(0.25).seed(2);
    for m in [&mut hard, &mut soft] {
        m.write("cat")?;
        m.write("dog")?;
        m.write_text("note", "meet at the harbour at nine")?;
        m.write("owl")?;
    }
    let mut run = Rng::new(RUN_SEED);
    o.show(hard.read("cat").address_noise(0.2).seed(1).stream(&mut run).answer());
    o.show(hard.read("note").address_noise(0.1).seed(2).stream(&mut run).answer());
    o.show(soft.read("cat").address_noise(0.2).seed(1).stream(&mut run).answer());
    o.show(soft.read("cat").address_noise(0.2).seed(1).mode(Mode::Settle).stream(&mut run).answer());
    o.show(hard.read_from_noise().seed(3).stream(&mut run).answer());
    o.show(hard.read("zebra").address_noise(0.0).seed(4).stream(&mut run).answer());
    o.show(soft.attend("cat").address_noise(0.2).stream(&mut run).answer());

    // the byte-counter memory, the travel rule and TRACK-C (no example program; the statements' docs carry them)
    o.line("== sdmscale");
    let mut k = SdmScale::build().name("k").word_size(256).hard_locations(20_000).activation_probability(0.01);
    k.put("cat")?;
    k.fill(200)?;
    o.show(k.read("cat").address_noise(0.2).seed(1).answer());
    o.show(k.read("cat").address_noise(0.2).via(Via::Pulls).wake(Wake::Top).seed(1).answer());
    o.show(k.read("owl").address_noise(0.0).seed(2).answer());
    o.line("== refusal");
    o.show(Refusal::build().word_size(256).load(3000).level(0.01).answer());
    o.line("== contenttrack");
    o.show(ContentTrack::build().word_size(256).hard_locations(20_000).load(300).address_noise(0.3).samples(50).answer());
    Ok(())
}

#[allow(dead_code)]
fn main() {
    print!("{}", run());
}
