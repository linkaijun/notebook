# (PART) 可视化 {.unnumbered}

# Manim动画 {#part6}

这一部分展示Manim动画成品。

## 最小二乘与投影矩阵 {#part6_1}

源码下载链接：


```{=html}
<a href="data:text/x-python;base64,ZnJvbSBtYW5pbSBpbXBvcnQgKg0KaW1wb3J0IG51bXB5IGFzIG5wDQoNCmNsYXNzIFByb2plY3Rpb24oTW92aW5nQ2FtZXJhU2NlbmUpOg0KICAgIGRlZiBjb25zdHJ1Y3Qoc2VsZik6DQogICAgICAgICMgUGFydCAxDQogICAgICAgICMg6K6+572u5Z2Q5qCH57O7DQogICAgICAgIHBsYW5lPUF4ZXMoDQogICAgICAgICAgICB4X3JhbmdlPVstMSwxMCwxXSwNCiAgICAgICAgICAgIHlfcmFuZ2U9Wy0xLDEwLDFdLA0KICAgICAgICAgICAgYXhpc19jb25maWc9eyJpbmNsdWRlX251bWJlcnMiOiBGYWxzZX0NCiAgICAgICAgKQ0KICAgICAgICBzZWxmLnBsYXkoQ3JlYXRlKHBsYW5lKSkNCg0KICAgICAgICAjIOeUn+aIkOaVo+eCueWvuQ0KICAgICAgICBucC5yYW5kb20uc2VlZCgwKQ0KICAgICAgICB4ID0gbnAuYXJhbmdlKDEsIDksIDEpDQogICAgICAgIG5vaXNlID0gbnAucmFuZG9tLnJhbmRuKDgpICogMg0KICAgICAgICB5ID0geCArIG5vaXNlDQogICAgICAgIHBvaW50cyA9IG5wLmNvbHVtbl9zdGFjaygoeCwgeSkpICAgIyDlkIjlubbmlbDnu4QNCiAgICAgICAgcG9pbnRzID0gVkdyb3VwKCpbRG90KHBsYW5lLmMycCgqcG9pbnQpKSBmb3IgcG9pbnQgaW4gcG9pbnRzXSkgICMg5bCG5pWj54K55omT5YyFDQogICAgICAgIHNlbGYucGxheShGYWRlSW4ocG9pbnRzLCBydW5fdGltZT0yKSkNCiAgICAgICAgc2VsZi53YWl0KDEpDQoNCiAgICAgICAgIyDnlJ/miJDlm57lvZLnur8NCiAgICAgICAgeF9tZWFuID0gbnAubWVhbih4KQ0KICAgICAgICB5X21lYW4gPSBucC5tZWFuKHkpDQogICAgICAgIHhfYmlhID0geCAtIHhfbWVhbg0KICAgICAgICB5X2JpYSA9IHkgLSB5X21lYW4NCiAgICAgICAgbF94eCA9IG5wLnN1bSh4X2JpYSAqKiAyKQ0KICAgICAgICBsX3h5ID0gbnAuc3VtKHhfYmlhICogeV9iaWEpDQogICAgICAgIGIgPSBsX3h5IC8gbF94eA0KICAgICAgICBhID0geV9tZWFuIC0gYiAqIHhfbWVhbg0KICAgICAgICByZWdyZXNzaW9uX2xpbmUgPSBwbGFuZS5wbG90KGxhbWJkYSB4OiBhK2IqeCwgY29sb3I9QkxVRSkNCiAgICAgICAgc2VsZi5wbGF5KFdyaXRlKHJlZ3Jlc3Npb25fbGluZSkpDQogICAgICAgIHRfMV8xID0gVGV4KHInJFxoYXR7eX1faSA9IFxoYXR7XGJldGFfMH0gKyBcaGF0e1xiZXRhXzF9IHhfaSQnKQ0KICAgICAgICB0XzFfMS5uZXh0X3RvKHJlZ3Jlc3Npb25fbGluZS5nZXRfZW5kKCksIExFRlQrVVApDQogICAgICAgIHNlbGYucGxheShXcml0ZSh0XzFfMSkpDQoNCiAgICAgICAgIyDmt7vliqDor6/lt64NCiAgICAgICAgeV9oYXQgPSBhK2IqeA0KICAgICAgICBlX2xpbmVfc3RhcnQgPSBucC5jb2x1bW5fc3RhY2soKHgsIHlfaGF0KSkNCiAgICAgICAgZV9saW5lX2VuZCA9IG5wLmNvbHVtbl9zdGFjaygoeCwgeSkpDQogICAgICAgIGVfbGluZXMgPSBWR3JvdXAoKQ0KICAgICAgICBmb3IgaSBpbiByYW5nZSgwLDgpOg0KICAgICAgICAgICAgZV9saW5lID0gTGluZShwbGFuZS5jMnAoKmVfbGluZV9zdGFydFtpXSksIHBsYW5lLmMycCgqZV9saW5lX2VuZFtpXSksIGNvbG9yPVJFRCkNCiAgICAgICAgICAgIGVfbGluZXMuYWRkKGVfbGluZSkNCiAgICAgICAgc2VsZi5wbGF5KEZhZGVJbihlX2xpbmVzLnNldF96X2luZGV4KC0xKSwgcnVuX3RpbWU9MikpICAjIHNldF96X2luZGV46K6+572u5Zu+5bGC6aG65bqPDQoNCiAgICAgICAgdF8xXzIgPSBCcmFjZUxhYmVsKGVfbGluZXNbM10sIHRleHQ9cidlX2k9eV9pIC0gXGhhdHt5X2l9JywgYnJhY2VfZGlyZWN0aW9uPW5wLmFycmF5KFstMSwwLDBdKSkgICMg6Iqx5ous5Y+35rOo6YeKDQogICAgICAgIHRfMV8yLnNldF9jb2xvcihSRUQpDQogICAgICAgIHNlbGYucGxheShXcml0ZSh0XzFfMikpDQoNCiAgICAgICAgIyDlnZDmoIfns7vnvKnlsI/lt6bnp7vvvIznu5nlj7PovrnnlZnlh7rnqbrpl7QNCiAgICAgICAgcGFydDEgPSBHcm91cChwbGFuZSwgcG9pbnRzLCByZWdyZXNzaW9uX2xpbmUsIGVfbGluZXMsIHRfMV8xLCB0XzFfMikNCiAgICAgICAgcGFydDEuZ2VuZXJhdGVfdGFyZ2V0KCkgICAjIOeUn+aIkOenu+WKqOWQjueahOacgOe7iOeKtuaAgQ0KICAgICAgICBwYXJ0MS50YXJnZXQuc2hpZnQoTEVGVCo0K0RPV04qMC4yKS5zY2FsZSgwLjUpDQogICAgICAgIHNlbGYucGxheShNb3ZlVG9UYXJnZXQocGFydDEsIHJ1bl90aW1lPTIpKQ0KDQogICAgICAgICMg5q2j6KeE5pa556iL57uEDQogICAgICAgIHRfMV8zID0gTWF0aFRleCgNCiAgICAgICAgICAgIHInJycNCiAgICAgICAgICAgIFEoXGhhdCBcYmV0YV8wLFxoYXQgXGJldGFfMSkmPVx1bmRlcnNldHtcaGF0IFxiZXRhXzAsIFxoYXQgXGJldGFfMX0ge1xhcmdcbWlufSBcc3VtXm5fe2k9MX0oeV9pLVxoYXQgeV9pKV4yXFwNCiAgICAgICAgICAgICY9XHVuZGVyc2V0e1xoYXQgXGJldGFfMCwgXGhhdCBcYmV0YV8xfSB7XGFyZ1xtaW59IFxzdW1ebl97aT0xfWVfaV4yIA0KICAgICAgICAgICAgJycnDQogICAgICAgICkNCiAgICAgICAgdF8xXzMuc2hpZnQoVVAqMS41K1JJR0hUKjMpLnNjYWxlKDAuNzUpDQogICAgICAgIHNlbGYucGxheShXcml0ZSh0XzFfMywgcnVuX3RpbWU9MikpDQogICAgICAgIHNlbGYud2FpdCgyKQ0KICAgICAgICB0XzFfNCA9IE1hdGhUZXgocicnJw0KICAgICAgICBcUmlnaHRhcnJvdyANCiAgICAgICAgXGxlZnRcew0KICAgICAgICBcYmVnaW57YXJyYXl9e2xsfQ0KICAgICAgICBcc3VtXGxpbWl0c197aT0xfV5uIGVfaT0wICBcXA0KICAgICAgICBcc3VtXGxpbWl0c197aT0xfV5uIHhfaWVfaT0wIA0KICAgICAgICBcZW5ke2FycmF5fQ0KICAgICAgICBccmlnaHQuDQogICAgICAgICcnJykNCiAgICAgICAgdF8xXzQubmV4dF90byh0XzFfMywgRE9XTikuc2NhbGUoMC43NSkNCiAgICAgICAgc2VsZi5wbGF5KFdyaXRlKHRfMV80LCBydW5fdGltZT0yKSkNCiAgICAgICAgc2VsZi53YWl0KDEpDQogICAgICAgICMgc2VsZi5hZGQoaW5kZXhfbGFiZWxzKHRfMV81WzBdKSkNCiAgICAgICAgc2VsZi5wbGF5KHRfMV80WzBdWzg6XS5hbmltYXRlLnNldF9jb2xvcihZRUxMT1cpKQ0KICAgICAgICBzZWxmLndhaXQoMikNCg0KICAgICAgICAjIOa4heeQhuWvueixoQ0KICAgICAgICBwYXJ0MV90ZXh0ID0gR3JvdXAodF8xXzMsIHRfMV80KQ0KICAgICAgICBzZWxmLnBsYXkoRmFkZU91dChwYXJ0MSksIEZhZGVPdXQocGFydDFfdGV4dCkpDQogICAgICAgIHNlbGYud2FpdCgxKQ0KDQogICAgICAgICMgcGFydDINCiAgICAgICAgIyDnn6npmLXlvaLlvI8NCiAgICAgICAgdF8yXzEgPSBNYXRoVGV4KCdZJywgJz0nLCByJycnDQogICAgICAgICAgICAgICAgICAgICAgICBcYmVnaW57Ym1hdHJpeH0NCiAgICAgICAgICAgICAgICAgICAgICAgIHlfMSBcXA0KICAgICAgICAgICAgICAgICAgICAgICAgeV8yIFxcDQogICAgICAgICAgICAgICAgICAgICAgICBcdmRvdHMgXFwNCiAgICAgICAgICAgICAgICAgICAgICAgIHlfbiANCiAgICAgICAgICAgICAgICAgICAgICAgIFxlbmR7Ym1hdHJpeH1fe24gXHRpbWVzIDF9DQogICAgICAgICAgICAgICAgICAgICAgICAnJycpDQoNCiAgICAgICAgdF8yXzIgPSBNYXRoVGV4KCdYJywgJz0nLCByJycnDQogICAgICAgICAgICAgICAgXGJlZ2lue2JtYXRyaXh9DQogICAgICAgICAgICAgICAgMSAmIHhfezExfSAmIFxjZG90cyAmIHhfezFwfSBcXA0KICAgICAgICAgICAgICAgIDEgJiB4X3syMX0gJiBcY2RvdHMgJiB4X3sycH0gXFwNCiAgICAgICAgICAgICAgICBcdmRvdHMgJiBcdmRvdHMgJiBcZGRvdHMgJiBcdmRvdHMgXFwNCiAgICAgICAgICAgICAgICAxICYgeF97bjF9ICYgXGNkb3RzICYgeF97bnB9DQogICAgICAgICAgICAgICAgXGVuZHtibWF0cml4fV97biBcdGltZXMgKHArMSl9DQogICAgICAgICAgICAgICAgJycnKQ0KDQogICAgICAgIHRfMl8zID0gTWF0aFRleChyJ1xiZXRhJywgJz0nLCByJycnDQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIFxiZWdpbntibWF0cml4fQ0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBcYmV0YV8wIFxcDQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIFxiZXRhXzEgXFwNCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgXHZkb3RzIFxcDQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIFxiZXRhX3ANCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgXGVuZHtibWF0cml4fV97KHArMSkgXHRpbWVzIDF9DQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICcnJykNCg0KICAgICAgICB0XzJfNCA9IE1hdGhUZXgocidcZXBzaWxvbicsICc9JywgcicnJw0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBcYmVnaW57Ym1hdHJpeH0NCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgXGVwc2lsb25fMSBcXA0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBcZXBzaWxvbl8yIFxcDQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIFx2ZG90cyBcXA0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBcZXBzaWxvbl9uIA0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBcZW5ke2JtYXRyaXh9X3tuIFx0aW1lcyAxfQ0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAnJycpDQoNCiAgICAgICAgc2VsZi5wbGF5KFdyaXRlKHRfMl8xLnNoaWZ0KExFRlQgKiA1KS5zY2FsZSgwLjYpKSwgV3JpdGUodF8yXzIubmV4dF90byh0XzJfMSwgUklHSFQsIGJ1ZmY9MC4wNSkuc2NhbGUoMC42KSksDQogICAgICAgICAgICAgICAgICBXcml0ZSh0XzJfMy5uZXh0X3RvKHRfMl8yLCBSSUdIVCwgYnVmZj0wLjA1KS5zY2FsZSgwLjYpKSwNCiAgICAgICAgICAgICAgICAgIFdyaXRlKHRfMl80Lm5leHRfdG8odF8yXzMsIFJJR0hULCBidWZmPTAuMDUpLnNjYWxlKDAuNikpKQ0KICAgICAgICBzZWxmLndhaXQoMikNCg0KICAgICAgICB0XzJfNSA9IE1hdGhUZXgoJ1knLCAnPScsICdYJywgcidcYmV0YScsICcrJywgcidcZXBzaWxvbicpLnNjYWxlKDEuNSkNCiAgICAgICAgdF8yXzEyMzQgPSBHcm91cCh0XzJfMSwgdF8yXzIsIHRfMl8zLCB0XzJfNCkNCiAgICAgICAgc2VsZi5wbGF5KFJlcGxhY2VtZW50VHJhbnNmb3JtKHRfMl8xMjM0LCB0XzJfNSkpDQogICAgICAgIHNlbGYud2FpdCgxKQ0KDQogICAgICAgICMg5byV5Ye65YiX56m66Ze0DQogICAgICAgIHNlbGYucGxheShJbmRpY2F0ZSh0XzJfNVs1XSwgY29sb3I9UkVEKSkgICAjIEluZGljYXRl6KGo5by66LCDDQogICAgICAgIHNlbGYud2FpdCgxKQ0KICAgICAgICBzZWxmLnBsYXkoRmFkZU91dCh0XzJfNVs0Ol0pKQ0KICAgICAgICBzZWxmLnBsYXkodF8yXzVbOjRdLmFuaW1hdGUubW92ZV90byhPUklHSU4pKQ0KICAgICAgICBzZWxmLnBsYXkoVHJhbnNmb3JtKHRfMl81WzFdLCBNYXRoVGV4KHInXG5lcScpLm1vdmVfdG8odF8yXzVbMV0pLnNldF9jb2xvcihSRUQpKSkNCiAgICAgICAgc2VsZi53YWl0KDEpDQogICAgICAgIHNlbGYucGxheShJbmRpY2F0ZSh0XzJfNVswXSwgY29sb3I9QkxVRSwgcnVuX3RpbWU9MiksIEluZGljYXRlKHRfMl81WzJdLCBjb2xvcj1CTFVFLCBydW5fdGltZT0yKSkNCiAgICAgICAgc2VsZi53YWl0KDEpDQogICAgICAgIHNlbGYucGxheShJbmRpY2F0ZSh0XzJfNVszXSwgY29sb3I9WUVMTE9XLCBydW5fdGltZT0yKSkNCiAgICAgICAgc2VsZi53YWl0KDEpDQogICAgICAgIHNlbGYucGxheShGYWRlT3V0KHRfMl81Wzo0XSkpDQogICAgICAgIHNlbGYud2FpdCgxKQ0KDQogICAgICAgIHRfMl82ID0gTWF0aFRleChyJ1hcYmV0YScsICc9JywgcicnJw0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBcYmVnaW57Ym1hdHJpeH0NCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgWF8wICYgWF8xICYgXGNkb3RzICYgWF9wIA0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBcZW5ke2JtYXRyaXh9DQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIFxiZWdpbntibWF0cml4fQ0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBcYmV0YV8wIFxcDQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIFxiZXRhXzEgXFwNCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgXHZkb3RzIFxcDQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIFxiZXRhX3AgDQogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIFxlbmR7Ym1hdHJpeH0NCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJycnLCAnPScsIHInWF8wXGJldGFfMCtYXzFcYmV0YV8xKyBcY2RvdHMgK1hfcFxiZXRhX3AnKS5zY2FsZSgwLjc1KQ0KICAgICAgICBzZWxmLnBsYXkoV3JpdGUodF8yXzYsIHJ1bl90aW1lPTIpKQ0KICAgICAgICBzZWxmLndhaXQoMikNCiAgICAgICAgdF8yXzcgPSBUZXh0KCdUaGUgY29sdW1uIHNwYWNlIG9mIFgnLCBmb250PSdHYWJyaW9sYScsIGZvbnRfc2l6ZT05NiwgZ3JhZGllbnQ9KEJMVUUsIFlFTExPVywgUkVEKSkuc2hpZnQoDQogICAgICAgICAgICBVUCAqIDAuNSkNCiAgICAgICAgdF8yXzdfc3ViID0gVGV4dCgnQyhYKScsIGdyYWRpZW50PShCTFVFLCBZRUxMT1csIFJFRCkpLm5leHRfdG8odF8yXzcsIERPV04pDQogICAgICAgIHRfMl83ID0gVkdyb3VwKHRfMl83LCB0XzJfN19zdWIpDQogICAgICAgIHNlbGYucGxheShSZXBsYWNlbWVudFRyYW5zZm9ybSh0XzJfNiwgdF8yXzcsIHJ1bl90aW1lPTEuNSkpDQogICAgICAgIHNlbGYucGxheShBcHBseVdhdmUodF8yXzcpKSAgICMg6K6p5paH5pys5rOi5rWq6LW35p2lfg0KICAgICAgICBzZWxmLndhaXQoMikNCiAgICAgICAgc2VsZi5wbGF5KEZhZGVPdXQodF8yXzcsIHJ1bl90aW1lPTIpKQ0KDQogICAgICAgIHRfMl84XzEgPSBNYXRoVGV4KHInWCBcYmV0YSBcLCBcaW4gXCwgQyhYKScpDQogICAgICAgIHRfMl84XzIgPSBNYXRoVGV4KHInXGhhdHtZfSBcLCA9IFwsIFggXGhhdHtcYmV0YX0nKQ0KICAgICAgICB0XzJfOF8zID0gTWF0aFRleChyJ1xoYXR7WX0gXHRvIFknKQ0KICAgICAgICB0XzJfOF8xMjMgPSBWR3JvdXAodF8yXzhfMSwgdF8yXzhfMiwgdF8yXzhfMykuYXJyYW5nZShET1dOKS5zY2FsZSgxLjUpDQogICAgICAgIHNlbGYucGxheShXcml0ZSh0XzJfOF8xMjMpKQ0KICAgICAgICBzZWxmLndhaXQoMSkNCiAgICAgICAgc2VsZi5wbGF5KEluZGljYXRlKHRfMl84XzEsIHJ1bl90aW1lPTIsIGNvbG9yPUJMVUUpKQ0KICAgICAgICBzZWxmLndhaXQoMSkNCiAgICAgICAgc2VsZi5wbGF5KEluZGljYXRlKHRfMl84XzIsIHJ1bl90aW1lPTIsIGNvbG9yPVlFTExPVykpDQogICAgICAgIHNlbGYud2FpdCgxKQ0KICAgICAgICBzZWxmLnBsYXkoSW5kaWNhdGUodF8yXzhfMywgcnVuX3RpbWU9MiwgY29sb3I9WUVMTE9XKSkNCiAgICAgICAgc2VsZi53YWl0KDIpDQogICAgICAgIHNlbGYucGxheSh0XzJfOF8yWzBdWzQ6XS5hbmltYXRlLnNldF9jb2xvcihZRUxMT1cpKQ0KICAgICAgICBzZWxmLndhaXQoMSkNCiAgICAgICAgc2VsZi5wbGF5KEZhZGVPdXQodF8yXzhfMSksIEZhZGVPdXQodF8yXzhfMlswXVswOjRdKSwgRmFkZU91dCh0XzJfOF8zKSkNCiAgICAgICAgc2VsZi5wbGF5KHRfMl84XzJbMF1bNDpdLmFuaW1hdGUubW92ZV90byhPUklHSU4pKQ0KICAgICAgICB0XzJfOSA9IFRleHQoJz8nLCBmb250PSdUaW1lcyBOZXcgUm9tYW4nKS5uZXh0X3RvKHRfMl84XzJbMF1bNDpdKQ0KICAgICAgICBzZWxmLnBsYXkoV3JpdGUodF8yXzkpKQ0KICAgICAgICBzZWxmLndhaXQoMSkNCiAgICAgICAgc2VsZi5wbGF5KEZhZGVPdXQodF8yXzhfMlswXVs0Ol0pLCBGYWRlT3V0KHRfMl85KSkNCiAgICAgICAgc2VsZi53YWl0KDEpDQoNCiAgICAgICAgIyBwYXJ0Mw0KICAgICAgICAjIOe7mOWItuW5s+mdog0KICAgICAgICBjeF9wb3NpdGlvbnMgPSBbDQogICAgICAgICAgICBbNSwgMSwgMF0sDQogICAgICAgICAgICBbMiwgLTIsIDBdLA0KICAgICAgICAgICAgWy02LCAtMiwgMF0sDQogICAgICAgICAgICBbLTMsIDEsIDBdDQogICAgICAgIF0NCiAgICAgICAgY3ggPSBQb2x5Z29uKCpjeF9wb3NpdGlvbnMsIGZpbGxfY29sb3I9TWFuaW1Db2xvcignIzU4QzRERCcpLCBmaWxsX29wYWNpdHk9MC41KSAgICMg57uY5Yi25bCB6Zet5Zu+5b2iDQogICAgICAgIHRfM18xID0gVGV4dCgnQyhYKScsIGNvbG9yPUJMVUUsIGZvbnRfc2l6ZT0zNikNCiAgICAgICAgdF8zXzEubmV4dF90byhjeCwgUklHSFQgKyBVUCkNCiAgICAgICAgc2VsZi5wbGF5KFdyaXRlKGN4KSkNCiAgICAgICAgc2VsZi5wbGF5KFdyaXRlKHRfM18xKSkNCiAgICAgICAgc2VsZi53YWl0KDEpDQoNCiAgICAgICAgIyDmt7vliqDlkJHph48NCiAgICAgICAgeV92ZWMgPSBBcnJvdyhzdGFydD1bLTMsIDAsIDBdLCBlbmQ9WzIsIDMsIDBdLCBjb2xvcj1XSElURSwgYnVmZj0wKQ0KICAgICAgICB0XzNfMiA9IFRleHQoJ1knLCBjb2xvcj1XSElURSwgZm9udF9zaXplPTM2KS5uZXh0X3RvKHRfM18xLCBET1dOKQ0KICAgICAgICBzZWxmLnBsYXkoV3JpdGUoeV92ZWMpLCBXcml0ZSh0XzNfMikpDQogICAgICAgIHNlbGYud2FpdCgxKQ0KDQogICAgICAgIHBvaW50X2xzID0gW1sxLCAtMS41LCAwXSwgWzMsIDAsIDBdLCBbMiwgLTEsIDBdXQ0KICAgICAgICB5aGF0X3ZlYyA9IEFycm93KHN0YXJ0PVstMywgMCwgMF0sIGVuZD1bLTEsIC0xLCAwXSwgY29sb3I9R1JFRU4sIGJ1ZmY9MCkNCiAgICAgICAgZV92ZWMgPSBBcnJvdyhzdGFydD1bLTEsIC0xLCAwXSwgZW5kPVsyLCAzLCAwXSwgY29sb3I9UkVELCBidWZmPTApDQogICAgICAgIHRfM18zID0gTWF0aFRleChyJ1xoYXR7WX09WFxoYXR7XGJldGF9JywgY29sb3I9R1JFRU4sIGZvbnRfc2l6ZT0zNikubmV4dF90byh0XzNfMiwgRE9XTikNCiAgICAgICAgdF8zXzQgPSBNYXRoVGV4KHInZT1ZLVxoYXR7WX0nLCBjb2xvcj1SRUQsIGZvbnRfc2l6ZT0zNikubmV4dF90byh0XzNfMywgRE9XTikNCiAgICAgICAgc2VsZi5wbGF5KFdyaXRlKHloYXRfdmVjKSwgV3JpdGUodF8zXzMpKQ0KICAgICAgICBzZWxmLnBsYXkoV3JpdGUoZV92ZWMpLCBXcml0ZSh0XzNfNCkpDQoNCiAgICAgICAgIyDorr7nva7lkJHph4/nmoTlj5jmjaLmg4XlhrUNCiAgICAgICAgZm9yIHBvaW50IGluIHBvaW50X2xzOg0KICAgICAgICAgICAgbmV3X3loYXRfdmVjID0gQXJyb3coc3RhcnQ9Wy0zLCAwLCAwXSwgZW5kPXBvaW50LCBjb2xvcj1HUkVFTiwgYnVmZj0wKQ0KICAgICAgICAgICAgbmV3X2VfdmVjID0gQXJyb3coc3RhcnQ9cG9pbnQsIGVuZD1bMiwgMywgMF0sIGNvbG9yPVJFRCwgYnVmZj0wKQ0KICAgICAgICAgICAgc2VsZi5wbGF5KFJlcGxhY2VtZW50VHJhbnNmb3JtKHloYXRfdmVjLCBuZXdfeWhhdF92ZWMsIHJ1bl90aW1lPTEuNSksDQogICAgICAgICAgICAgICAgICAgICAgUmVwbGFjZW1lbnRUcmFuc2Zvcm0oZV92ZWMsIG5ld19lX3ZlYywgcnVuX3RpbWU9MS41KSkNCiAgICAgICAgICAgIHNlbGYud2FpdCgxKQ0KICAgICAgICAgICAgeWhhdF92ZWMgPSBuZXdfeWhhdF92ZWMNCiAgICAgICAgICAgIGVfdmVjID0gbmV3X2VfdmVjDQoNCiAgICAgICAgIyDnlLvlnoLotrPvvIzlsLHmmK/kuKTmnaHnur8NCiAgICAgICAgcm9vdF8xID0gTGluZShucC5hcnJheShbMS4wNSwgLTAuOCwgMF0pLCBucC5hcnJheShbMS4wNSwgMCwgMF0pLCBjb2xvcj1XSElURSkNCiAgICAgICAgcm9vdF8yID0gTGluZShucC5hcnJheShbMSwgMCwgMF0pLCBucC5hcnJheShbMiwgLTAuMiwgMF0pLCBjb2xvcj1XSElURSwgYnVmZj0wLjA1KQ0KICAgICAgICByb290ID0gVkdyb3VwKHJvb3RfMSwgcm9vdF8yKQ0KICAgICAgICAjIOWKqOeUu+acieWFiOWQjumhuuW6j++8jOWJjeS4gOS4quWujOaIkOWGjeaOpeS4i+S4gOS4qg0KICAgICAgICBzZWxmLnBsYXkoU3VjY2Vzc2lvbihXcml0ZShyb290XzEsIHJ1bl90aW1lPTAuMjUpLCBXcml0ZShyb290XzIsIHJ1bl90aW1lPTAuMjUpKSkNCiAgICAgICAgdF8zXzUgPSBNYXRoVGV4KHInXGhhdHtZfV97b2xzfT1YXGhhdHtcYmV0YX1fe29sc30nLCBjb2xvcj1ZRUxMT1csIGZvbnRfc2l6ZT0zNikubW92ZV90byh0XzNfMykNCiAgICAgICAgc2VsZi5wbGF5KFJlcGxhY2VtZW50VHJhbnNmb3JtKHRfM18zLCB0XzNfNSkpDQogICAgICAgIHNlbGYucGxheSh5aGF0X3ZlYy5hbmltYXRlLnNldF9jb2xvcihZRUxMT1cpKQ0KICAgICAgICBzZWxmLndhaXQoMikNCg0KICAgICAgICAjIHBhcnQ0DQogICAgICAgICMg5Yeg5L2V5Zu+5bem56e75LiU57yp5bCPDQogICAgICAgIHBhcnQzXzEgPSBWR3JvdXAoY3gsIHlfdmVjLCB5aGF0X3ZlYywgZV92ZWMsIHJvb3QpDQogICAgICAgIHBhcnQzXzIgPSBWR3JvdXAodF8zXzEsIHRfM18yLCB0XzNfNCwgdF8zXzUpDQogICAgICAgIHBhcnQzXzEuZ2VuZXJhdGVfdGFyZ2V0KCkNCiAgICAgICAgcGFydDNfMi5nZW5lcmF0ZV90YXJnZXQoKQ0KICAgICAgICBwYXJ0M18xLnRhcmdldC5zaGlmdChMRUZUICogMy41ICsgRE9XTiAqIDAuMikuc2NhbGUoMC42KQ0KICAgICAgICBwYXJ0M18yLnRhcmdldC5hcnJhbmdlX2luX2dyaWQobl9yb3dzPTIsIG5jb2xzPTMpLnNoaWZ0KExFRlQgKiA5ICsgVVAgKiAyLjUpLnNjYWxlKDAuNykNCiAgICAgICAgc2VsZi5wbGF5KE1vdmVUb1RhcmdldChwYXJ0M18xLCBydW5fdGltZT0yKSwgTW92ZVRvVGFyZ2V0KHBhcnQzXzIsIHJ1bl90aW1lPTIpKQ0KICAgICAgICBzZWxmLndhaXQoMSkNCiAgICAgICAgc2VsZi5wbGF5KEZhZGVPdXQoeV92ZWMpLCBGYWRlT3V0KHloYXRfdmVjKSkNCiAgICAgICAgc2VsZi53YWl0KDEpDQoNCiAgICAgICAgIyDmraPkuqTnrYnku7fkuo7mqKHplb/mnIDlsI/vvIznrYnku7fkuo7mrovlt67lubPmlrnlkozmnIDlsI8NCiAgICAgICAgdF80XzEgPSBNYXRoVGV4KHInZSBcLCBcYm90IFwsIEMoWCknKS5zaGlmdChVUCAqIDIgKyBSSUdIVCAqIDMpDQogICAgICAgIHNlbGYucGxheShXcml0ZSh0XzRfMSkpDQogICAgICAgIHNlbGYud2FpdCgxKQ0KICAgICAgICBzZWxmLnBsYXkoRmFkZUluKHlfdmVjKSwgRmFkZUluKHloYXRfdmVjKSkNCiAgICAgICAgc2VsZi53YWl0KDEpDQogICAgICAgIHRfNF8yID0gTWF0aFRleChyJycnDQogICAgICAgICAgICAgICAgJlxMZWZ0cmlnaHRhcnJvdyBcbWluIHx8ZXx8XzIgXFwNCiAgICAgICAgICAgICAgICAmXExlZnRyaWdodGFycm93IFxtaW4gUShcYmV0YSk9KFktWFxiZXRhKScoWS1YXGJldGEpXFwNCiAgICAgICAgICAgICAgICAmXExlZnRyaWdodGFycm93IFxtaW4gXHN1bVxsaW1pdHNfe2k9MX1ebiBlX2leMg0KICAgICAgICAgICAgICAgICcnJykuc2NhbGUoMC44KQ0KICAgICAgICBzZWxmLnBsYXkoV3JpdGUodF80XzIubmV4dF90byh0XzRfMSwgRE9XTiksIHJ1bl90aW1lPTIpKQ0KICAgICAgICBzZWxmLndhaXQoMSkNCiAgICAgICAgcGFydDFfdGV4dC5uZXh0X3RvKHRfNF8yLCBSSUdIVCo2KQ0KICAgICAgICBwYXJ0MV90ZXh0LnNjYWxlKDEuMikNCiAgICAgICAgcGFydDFfdGV4dFsxXS5zZXRfY29sb3IoV0hJVEUpDQogICAgICAgIHNlbGYuYWRkKHBhcnQxX3RleHQpDQogICAgICAgICMgc2VsZi5jYW1lcmEuZnJhbWUuYW5pbWF0ZS5zaGlmdOenu+WKqOinhuinku+8jOWPquiDveWcqE1vdmluZ0NhbWVyYVNjZW5l57G75LiL5L2/55SoDQogICAgICAgIHNlbGYucGxheShwYXJ0M18xLmFuaW1hdGUuc2hpZnQoTEVGVCksIHNlbGYuY2FtZXJhLmZyYW1lLmFuaW1hdGUuc2hpZnQoUklHSFQgKiA3KS5zZXQod2lkdGg9MTYpKQ0KICAgICAgICBzZWxmLndhaXQoMSkNCiAgICAgICAgIyBzZWxmLmFkZChpbmRleF9sYWJlbHMocGFydDFfdGV4dFswXVswXSksIGluZGV4X2xhYmVscyhwYXJ0MV90ZXh0WzFdWzBdKSwgaW5kZXhfbGFiZWxzKHRfNF8yWzBdKSkNCiAgICAgICAgc2VsZi5wbGF5KFN1Y2Nlc3Npb24oSW5kaWNhdGUodF8xXzNbMF1bNTI6XSksIEluZGljYXRlKHRfNF8yWzBdWzM2Ol0pKSkNCiAgICAgICAgc2VsZi53YWl0KDIpDQogICAgICAgIHNlbGYucGxheShwYXJ0M18xLmFuaW1hdGUuc2hpZnQoUklHSFQpLCBzZWxmLmNhbWVyYS5mcmFtZS5hbmltYXRlLnNoaWZ0KExFRlQgKiA3KS5zZXQod2lkdGg9MTQpKQ0KICAgICAgICBzZWxmLndhaXQoMSkNCiAgICAgICAgc2VsZi5wbGF5KEZhZGVPdXQodF80XzIpKQ0KICAgICAgICBzZWxmLndhaXQoMSkNCg0KICAgICAgICAjIOato+S6pA0KICAgICAgICB0XzRfMyA9IE1hdGhUZXgocicnJw0KICAgICAgICAgICAgICAgICZcTGVmdHJpZ2h0YXJyb3cgZSBcLCBcYm90IFwsIFhfaSBcLCAsIFw7IGk9MCwxLFxjZG90cyxuIFxcDQogICAgICAgICAgICAgICAgJlxMZWZ0cmlnaHRhcnJvdyBYX2knZT0wLCBcOyBpPTAsMSxcY2RvdHMsbiBcXA0KICAgICAgICAgICAgICAgICZcTGVmdHJpZ2h0YXJyb3cgWCdlPTANCiAgICAgICAgICAgICAgICAnJycpDQogICAgICAgIHNlbGYucGxheShXcml0ZSh0XzRfMy5uZXh0X3RvKHRfNF8xLCBET1dOKSwgcnVuX3RpbWU9MikpDQogICAgICAgIHNlbGYud2FpdCgyKQ0KICAgICAgICBzZWxmLnBsYXkocGFydDNfMS5hbmltYXRlLnNoaWZ0KExFRlQpLCBzZWxmLmNhbWVyYS5mcmFtZS5hbmltYXRlLnNoaWZ0KFJJR0hUICogNykuc2V0KHdpZHRoPTE2KSkNCiAgICAgICAgc2VsZi53YWl0KDEpDQogICAgICAgICMgc2VsZi5hZGQoaW5kZXhfbGFiZWxzKHRfNF8zWzBdKSkNCiAgICAgICAgc2VsZi5wbGF5KFN1Y2Nlc3Npb24oSW5kaWNhdGUodF8xXzRbMF1bODpdKSwgSW5kaWNhdGUodF80XzNbMF1bMzc6XSkpKQ0KICAgICAgICBzZWxmLndhaXQoMikNCiAgICAgICAgc2VsZi5wbGF5KHBhcnQzXzEuYW5pbWF0ZS5zaGlmdChSSUdIVCksIHNlbGYuY2FtZXJhLmZyYW1lLmFuaW1hdGUuc2hpZnQoTEVGVCAqIDcpLnNldCh3aWR0aD0xNCkpDQogICAgICAgIHNlbGYud2FpdCgyKQ0KDQogICAgICAgIHRfNF80ID0gTWF0aFRleChyJycnDQogICAgICAgICAgICAgICAgWCdlIFwsICY9IFwsIDAgXFwNCiAgICAgICAgICAgICAgICBYJyhZLVxoYXR7WX0pIFwsICY9IFwsIDAgXFwNCiAgICAgICAgICAgICAgICBYJyhZLVhcaGF0e1xiZXRhfSkgXCwgJj0gXCwgMCBcXA0KICAgICAgICAgICAgICAgIFgnWS1YJ1ggXGhhdHtcYmV0YX0gXCwgJj0gXCwgMCBcXA0KICAgICAgICAgICAgICAgIFxoYXR7XGJldGF9IFwsICY9IFwsIChYJ1gpXnstMX1YJ1kNCiAgICAgICAgICAgICAgICAnJycpLm5leHRfdG8odF80XzEsIERPV04pLnNjYWxlKDAuOSkNCiAgICAgICAgc2VsZi5wbGF5KFJlcGxhY2VtZW50VHJhbnNmb3JtKHRfNF8zLCB0XzRfNCkpDQogICAgICAgIHNlbGYud2FpdCgyKQ0KICAgICAgICAjIHNlbGYuYWRkKGluZGV4X2xhYmVscyh0XzRfNFswXSkpDQogICAgICAgIHNlbGYucGxheShTdWNjZXNzaW9uKENpcmN1bXNjcmliZSh0XzRfNFswXVszNzpdLCBzaGFwZT1SZWN0YW5nbGUpLA0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0XzRfNFswXVszNzpdLmFuaW1hdGUuc2V0X2NvbG9yKFlFTExPVykpKQ0KICAgICAgICBzZWxmLndhaXQoMikNCiAgICAgICAgc2VsZi5wbGF5KEZhZGVPdXQodF80XzEpLCBGYWRlT3V0KHRfNF80WzBdWzozN10pKQ0KICAgICAgICBzZWxmLnBsYXkodF80XzRbMF1bMzc6XS5hbmltYXRlLm1vdmVfdG8odF80XzEpKQ0KICAgICAgICBzZWxmLndhaXQoMSkNCg0KICAgICAgICAjIHBhcnQ1DQogICAgICAgICMg5oqV5b2x55+p6Zi1DQogICAgICAgIHRfNV8xID0gTWF0aFRleChyJycnDQogICAgICAgICAgICAgICAgXGhhdHtZfSBcLCAmPSBcLCBYXGhhdHtcYmV0YX0gXFwNCiAgICAgICAgICAgICAgICAmPSBcLCBYKFgnWCleey0xfVgnWQ0KICAgICAgICAgICAgICAgICcnJykuc2NhbGUoMC45KQ0KICAgICAgICAjIHNlbGYuYWRkKGluZGV4X2xhYmVscyh0XzVfMVswXSkpDQogICAgICAgIHRfNV8xWzBdWzQ6Nl0uc2V0X2NvbG9yKFlFTExPVykNCiAgICAgICAgc2VsZi5wbGF5KFdyaXRlKHRfNV8xLm5leHRfdG8odF80XzRbMF1bMzc6XSwgRE9XTiwgYWxpZ25lZF9lZGdlPUxFRlQpKSkNCiAgICAgICAgc2VsZi53YWl0KDEpDQoNCiAgICAgICAgdF81XzIgPSBCcmFjZUxhYmVsKHRfNV8xWzBdWzc6MTddLCAnSDpcLCBQcm9qZWN0aW9uIFwsIE1hdHJpeCcpLnNldF9jb2xvcihZRUxMT1cpDQogICAgICAgIHRfNV8zID0gTWF0aFRleChyIkgnXCwgPSBcLEgsIFwsIEheMiBcLCA9IFwsIEgiLCBjb2xvcj1ZRUxMT1cpLm5leHRfdG8odF81XzIsIERPV04pDQogICAgICAgIHNlbGYucGxheShTdWNjZXNzaW9uKFdyaXRlKHRfNV8yKSwgV3JpdGUodF81XzMpKSkNCiAgICAgICAgc2VsZi53YWl0KDIpDQogICAgICAgIHRfNV80ID0gTWF0aFRleChyJ1wsIEhZJywgc3Vic3RyaW5nc190b19pc29sYXRlPSdIJykuc2V0X2NvbG9yX2J5X3RleCgnSCcsIFlFTExPVykuc2NhbGUoMC45KS5tb3ZlX3RvKA0KICAgICAgICAgICAgdF81XzFbMF1bMzo2XSkNCiAgICAgICAgdF81XzFfc3ViID0gR3JvdXAodF81XzFbMF1bMzpdLCB0XzVfMiwgdF81XzMpDQogICAgICAgIHNlbGYucGxheShSZXBsYWNlbWVudFRyYW5zZm9ybSh0XzVfMV9zdWIsIHRfNV80KSkNCiAgICAgICAgc2VsZi53YWl0KDEpDQoNCiAgICAgICAgdF81XzQgPSBNYXRoVGV4KHInJycNCiAgICAgICAgICAgICAgICBlIFwsICY9IFwsIFktXGhhdHtZfSBcXA0KICAgICAgICAgICAgICAgICY9IFwsIFktSFkgXFwNCiAgICAgICAgICAgICAgICAmPSBcLCAoSS1IKVkgDQogICAgICAgICAgICAgICAgJycnKS5zY2FsZSgwLjkpLm5leHRfdG8odF81XzFbMF1bOjNdLCBET1dOLCBhbGlnbmVkX2VkZ2U9TEVGVCkNCiAgICAgICAgdF81XzRbMF1bMTM6MTZdLnNldF9jb2xvcihSRUQpDQogICAgICAgIHNlbGYucGxheShXcml0ZSh0XzVfNCkpDQogICAgICAgIHNlbGYud2FpdCgxKQ0KICAgICAgICB0XzVfNSA9IE1hdGhUZXgocidcLCAoSS1IKVknKS5zY2FsZSgwLjkpLm5leHRfdG8odF81XzRbMF1bOjJdLCBSSUdIVCkNCiAgICAgICAgdF81XzVbMF1bMTo0XS5zZXRfY29sb3IoUkVEKQ0KICAgICAgICBzZWxmLnBsYXkoUmVwbGFjZW1lbnRUcmFuc2Zvcm0odF81XzRbMF1bMjpdLCB0XzVfNSkpDQogICAgICAgIHNlbGYud2FpdCgyKQ0KDQogICAgICAgIHRfNV82ID0gTWF0aFRleChyJ0goSS1IKSBcLCA9IFwsIEgtSF4yIFwsID0gXCwgMCcpLm5leHRfdG8odF81XzQsIERPV04gKiAyKQ0KICAgICAgICB0XzVfNlswXVswXS5zZXRfY29sb3IoWUVMTE9XKQ0KICAgICAgICB0XzVfNlswXVsyOjVdLnNldF9jb2xvcihSRUQpDQogICAgICAgIHNlbGYucGxheShXcml0ZSh0XzVfNikpDQogICAgICAgIHNlbGYucGxheShBcHBseVdhdmUodF81XzYpKQ0KICAgICAgICBzZWxmLndhaXQoMykNCg0KICAgICAgICAjIOiwouW5lQ0KICAgICAgICBzZWxmLnBsYXkoKltGYWRlT3V0KG1vYikgZm9yIG1vYiBpbiBzZWxmLm1vYmplY3RzXSkgIyBzZWxmLm1vYmplY3Rz5a2Y5YKo5LqG5b2T5YmN6aG16Z2i55qE5omA5pyJ5a+56LGhDQogICAgICAgIHNlbGYud2FpdCgxKQ0KICAgICAgICB0X2VuZF8xID0gVGV4dCgnVGhhbmsgeW91IGZvciB3YXRjaGluZycsIGZvbnRfc2l6ZT0xMzAsIGZvbnQ9J0Vkd2FyZGlhbiBTY3JpcHQgSVRDJykuc2hpZnQoVVAgKiAwLjUpDQogICAgICAgIHRfZW5kXzIgPSBUZXh0KCdQcm9kdWNlZCBieSBsa2onLCBmb250X3NpemU9NDAsIGZvbnQ9J1NlZ29lIFNjcmlwdCcpLm5leHRfdG8odF9lbmRfMSwgRE9XTikNCiAgICAgICAgc2VsZi5wbGF5KFN1Y2Nlc3Npb24oV3JpdGUodF9lbmRfMSwgcnVuX3RpbWU9MiksIFdyaXRlKHRfZW5kXzIsIHJ1bl90aW1lPTEpKSkNCiAgICAgICAgc2VsZi53YWl0KDEuNSkNCg0KDQoNCg0KDQoNCg0KDQoNCg0KDQoNCg0KDQoNCg0KDQoNCg0K" download="projection.py">最小二乘与投影矩阵源码</a>
```

视频链接：[几何视角](#part3_4_3) & [B站：最小二乘与投影矩阵](https://www.bilibili.com/video/BV1eFxMeKEpM/)
