"""Test of the URL cleaning (MTA 3.9.1): fast on pathological texts, same result as before.

Run: python test_urls.py
"""
import re
import time

import mta_core as mta

ok = True


def check(cond, msg):
    global ok
    print(("✓ " if cond else "✗ ") + msg)
    ok = ok and cond


# the pattern of MTA up to 3.9, kept here only to compare results on ordinary texts
OLD = (r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)'''
       r'''(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+'''
       r'''(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|'''
       r'''[^\s`!()\[\]{};:'".,<>?«»""''])'''
       r''')''')

texts = ["see https://example.com/page?x=1&y=2.", "www.uni-halle.de (Soziologie)",
         "link (https://en.wikipedia.org/wiki/Foo_(bar)) here", "mail me at a.b@c.de, thanks",
         "go to shop.example.org/items/ now!", "no url at all", "http://x.com/(a(b)c)d end",
         "visit www2.site.net/path/to/file.html; then", "https://a.b/" + "x" * 300]
check(mta._remove_urls(texts) == [re.sub(OLD, '', t) for t in texts], "same result as before on ordinary texts")

bad = ["https://" + "." * 60 + " rest", "www.site.com/" + "," * 80, "see http://x.org/" + "!" * 50]
t0 = time.time()
out = mta._remove_urls(bad)
check(time.time() - t0 < 1.0, f"URL followed by a run of punctuation: {time.time() - t0:.3f} s (before: hours)")
check(out[0].endswith(" rest"), "the text after the URL is kept")

t0 = time.time()
mta._remove_urls(["https://site.com/" + "," * 40000])
check(time.time() - t0 < 1.0, "40 000 characters after a URL: under a second")

print("✓ all tests passed" if ok else "✗ some tests failed")
raise SystemExit(0 if ok else 1)
