/* ============================================================
   fx.js — The Dark Side of the Data
   A Pink Floyd song classifier rendered as the site background:
   a feed-forward neural network you can feed songs into and
   watch the signal propagate to a verdict.
   Plus: a realistic ECG heartbeat, tilt cards, stat counters.
   ============================================================ */
(function () {
    'use strict';

    var reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    var finePointer = window.matchMedia('(pointer: fine)').matches;

    // Muted Floyd palette: rust, amber, steel blue
    var SPECTRUM = ['#b35433', '#d98e3f', '#5b84a8'];
    var AMBER = '#d98e3f', STEEL = '#5b84a8';
    var YES_COLOR = '#4a9d6b', NO_COLOR = '#bf4a35';

    /* ------------------------------------------------------------
       The network: 6 audio features → 8 → 6 → 1 (is it Floyd?)
       ------------------------------------------------------------ */
    var LAYERS = [6, 8, 6, 1];
    var FEATURES = ['waveform', 'tempo', 'guitar', 'synths', 'vocals', 'ambience'];

    var SONGS = [
        { name: 'Time', artist: 'Pink Floyd', floyd: true, f: [0.90, 0.55, 0.95, 0.80, 0.70, 0.60] },
        { name: 'Comfortably Numb', artist: 'Pink Floyd', floyd: true, f: [0.85, 0.50, 1.00, 0.75, 0.80, 0.45] },
        { name: 'Wish You Were Here', artist: 'Pink Floyd', floyd: true, f: [0.70, 0.55, 0.90, 0.35, 0.85, 0.30] },
        { name: 'The Great Gig in the Sky', artist: 'Pink Floyd', floyd: true, f: [0.80, 0.45, 0.50, 0.70, 0.95, 0.90] },
        { name: 'Bohemian Rhapsody', artist: 'Queen', floyd: false, f: [0.95, 0.90, 0.60, 0.50, 0.98, 0.40] },
        { name: 'Stairway to Heaven', artist: 'Led Zeppelin', floyd: false, f: [0.75, 0.70, 0.90, 0.25, 0.80, 0.20] },
        { name: 'Paranoid Android', artist: 'Radiohead', floyd: false, f: [0.90, 0.85, 0.80, 0.70, 0.75, 0.30] },
        { name: 'Hotel California', artist: 'Eagles', floyd: false, f: [0.70, 0.60, 0.85, 0.30, 0.80, 0.25] }
    ];

    var YES_LINES = [
        '✓ YES — "Shine on, you crazy diamond."',
        '✓ YES — "Welcome, my son, welcome to the machine."',
        '✓ YES — "Breathe, breathe in the air… it\'s Floyd."'
    ];
    var NO_LINES = [
        '✗ NO — "There\'s someone in my head, but it\'s not Floyd."',
        '✗ NO — "How I wish, how I wish you were Floyd."',
        '✗ NO — "Just another brick in someone else\'s wall."'
    ];

    // Deterministic pseudo-random weights (seeded, so the "model" is stable)
    function mulberry32(seed) {
        return function () {
            seed |= 0; seed = seed + 0x6D2B79F5 | 0;
            var t = Math.imul(seed ^ seed >>> 15, 1 | seed);
            t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
            return ((t ^ t >>> 14) >>> 0) / 4294967296;
        };
    }
    var rand = mulberry32(1973); // the year the dark side of the moon landed

    var weights = [];
    for (var l = 0; l < LAYERS.length - 1; l++) {
        var wl = [];
        for (var i = 0; i < LAYERS[l]; i++) {
            var row = [];
            for (var j = 0; j < LAYERS[l + 1]; j++) row.push(rand() * 2 - 1);
            wl.push(row);
        }
        weights.push(wl);
    }

    function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

    // Real forward pass for the hidden layers; the output layer is
    // "well trained" (i.e. we know the answer).
    function forward(song) {
        var acts = [song.f.slice()];
        for (var l = 0; l < LAYERS.length - 1; l++) {
            var next = [];
            for (var j = 0; j < LAYERS[l + 1]; j++) {
                var sum = 0;
                for (var i = 0; i < LAYERS[l]; i++) sum += acts[l][i] * weights[l][i][j];
                next.push(sigmoid(sum * 1.6));
            }
            acts.push(next);
        }
        acts[LAYERS.length - 1][0] = song.floyd ? 0.96 : 0.06;
        return acts;
    }

    /* ------------------------------------------------------------
       Background canvas — the living network
       ------------------------------------------------------------ */
    var canvas = document.getElementById('bg-canvas');
    var verdictEl = document.getElementById('classifier-verdict');
    var chipsEl = document.getElementById('classifier-chips');

    var LAYER_DELAY = 560;   // ms between layers firing
    var PULSE_DUR = 560;     // ms for a pulse to cross an edge
    var TOTAL = LAYER_DELAY * (LAYERS.length - 2) + PULSE_DUR; // time to verdict

    var mouse = { x: -9999, y: -9999 };
    addEventListener('mousemove', function (e) {
        mouse.x = e.clientX; mouse.y = e.clientY;
    }, { passive: true });

    if (canvas) initNetwork();

    function initNetwork() {
        var ctx = canvas.getContext('2d');
        var dpr = Math.min(devicePixelRatio || 1, 2);
        var W = 0, H = 0;
        var nodes = []; // nodes[l][i] = {x, y, a (displayed activation)}

        function layout() {
            W = innerWidth; H = innerHeight;
            canvas.width = W * dpr;
            canvas.height = H * dpr;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

            var left = (W > 900 ? 250 : 0) + 90;
            var right = W - 130;
            nodes = [];
            for (var l = 0; l < LAYERS.length; l++) {
                var col = [];
                var x = left + (right - left) * (l / (LAYERS.length - 1));
                var n = LAYERS[l];
                var top = 90, bottom = H - 120;
                for (var i = 0; i < n; i++) {
                    var y = n === 1
                        ? (top + bottom) / 2
                        : top + (bottom - top) * (i + 0.5) / n;
                    col.push({ x: x, y: y, a: 0.08 });
                }
                nodes.push(col);
            }
        }
        layout();
        addEventListener('resize', layout);

        // Propagation state
        var prop = null; // {t0, acts, song, verdictShown}

        function classify(song, instant) {
            prop = {
                t0: performance.now(),
                acts: forward(song),
                song: song,
                verdictShown: false,
                instant: !!instant
            };
            if (verdictEl) {
                verdictEl.textContent = 'listening… signal propagating through the layers';
                verdictEl.className = 'classifier-verdict thinking';
            }
        }

        function showVerdict(song) {
            if (!verdictEl) return;
            var pool = song.floyd ? YES_LINES : NO_LINES;
            verdictEl.textContent = pool[Math.floor(Math.random() * pool.length)];
            verdictEl.className = 'classifier-verdict ' + (song.floyd ? 'yes' : 'no');
        }

        // Song chips (only exist on the home page)
        var lastUserAction = 0;
        var liveTimer = null;
        if (chipsEl) {
            SONGS.forEach(function (song) {
                var b = document.createElement('button');
                b.className = 'song-chip';
                b.type = 'button';
                b.innerHTML = song.name + ' <span class="chip-artist">?</span>';
                b.addEventListener('click', function () {
                    lastUserAction = performance.now();
                    // lift the scrim so the propagation plays at full brightness
                    if (!reduced) {
                        document.body.classList.add('net-live');
                        clearTimeout(liveTimer);
                        liveTimer = setTimeout(function () {
                            document.body.classList.remove('net-live');
                        }, TOTAL + 2600);
                    }
                    chipsEl.querySelectorAll('.song-chip').forEach(function (c) {
                        c.classList.remove('active');
                        c.querySelector('.chip-artist').textContent = '?';
                    });
                    b.classList.add('active');
                    // reveal the artist once the network answers
                    setTimeout(function () {
                        b.querySelector('.chip-artist').textContent = song.artist;
                    }, reduced ? 0 : TOTAL);
                    classify(song, reduced);
                });
                chipsEl.appendChild(b);
            });
        }

        // Ambient mode: the network keeps listening on its own
        if (!reduced) {
            setInterval(function () {
                if (performance.now() - lastUserAction > 8500) {
                    classify(SONGS[Math.floor(Math.random() * SONGS.length)]);
                }
            }, 9000);
            classify(SONGS[0]);
        } else {
            classify(SONGS[0], true);
        }

        function transitionProgress(l, now) {
            // progress (0..1) of the pulse crossing transition l (layer l -> l+1)
            if (!prop) return 0;
            if (prop.instant) return 1;
            var t = now - prop.t0 - l * LAYER_DELAY;
            return Math.max(0, Math.min(1, t / PULSE_DUR));
        }

        function draw(now) {
            ctx.clearRect(0, 0, W, H);
            ctx.globalCompositeOperation = 'lighter';

            var l, i, j, a, b_;

            // --- edges + travelling pulses ---
            for (l = 0; l < LAYERS.length - 1; l++) {
                var p = transitionProgress(l, now);
                for (i = 0; i < LAYERS[l]; i++) {
                    a = nodes[l][i];
                    for (j = 0; j < LAYERS[l + 1]; j++) {
                        b_ = nodes[l + 1][j];
                        var w = weights[l][i][j];
                        var srcAct = prop ? prop.acts[l][i] : 0;
                        var lit = p > 0 && p < 1 ? srcAct * Math.abs(w) : 0;

                        ctx.strokeStyle = 'rgba(155,160,190,' + (0.03 + 0.06 * Math.abs(w) + lit * 0.3) + ')';
                        ctx.lineWidth = 0.5 + Math.abs(w) * 1.0;
                        ctx.beginPath();
                        ctx.moveTo(a.x, a.y);
                        ctx.lineTo(b_.x, b_.y);
                        ctx.stroke();

                        // the signal itself
                        if (p > 0 && p < 1 && srcAct * Math.abs(w) > 0.12) {
                            var px = a.x + (b_.x - a.x) * p;
                            var py = a.y + (b_.y - a.y) * p;
                            var g = ctx.createRadialGradient(px, py, 0, px, py, 7);
                            var col = w > 0 ? AMBER : STEEL;
                            g.addColorStop(0, col);
                            g.addColorStop(1, 'rgba(0,0,0,0)');
                            ctx.fillStyle = g;
                            ctx.globalAlpha = Math.min(1, srcAct * Math.abs(w) * 1.6);
                            ctx.beginPath();
                            ctx.arc(px, py, 7, 0, Math.PI * 2);
                            ctx.fill();
                            ctx.globalAlpha = 1;
                        }
                    }
                }
            }

            // --- nodes ---
            for (l = 0; l < LAYERS.length; l++) {
                var fired = !prop ? false
                    : prop.instant ? true
                        : now - prop.t0 >= (l === 0 ? 0 : (l - 1) * LAYER_DELAY + PULSE_DUR);
                for (i = 0; i < LAYERS[l]; i++) {
                    var nd = nodes[l][i];
                    var target = prop && fired ? 0.15 + prop.acts[l][i] * 0.85 : 0.08;
                    nd.a += (target - nd.a) * 0.12;

                    var isOut = l === LAYERS.length - 1;
                    var baseR = isOut ? 14 : 5.5;
                    var color = isOut && prop && fired
                        ? (prop.song.floyd ? YES_COLOR : NO_COLOR)
                        : '#cdd2ea';

                    var g2 = ctx.createRadialGradient(nd.x, nd.y, 0, nd.x, nd.y, baseR * 3.4);
                    g2.addColorStop(0, color);
                    g2.addColorStop(1, 'rgba(0,0,0,0)');
                    ctx.fillStyle = g2;
                    ctx.globalAlpha = 0.10 + nd.a * 0.6;
                    ctx.beginPath();
                    ctx.arc(nd.x, nd.y, baseR * 3.4, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.globalAlpha = 1;

                    ctx.fillStyle = color;
                    ctx.globalAlpha = 0.28 + nd.a * 0.6;
                    ctx.beginPath();
                    ctx.arc(nd.x, nd.y, baseR, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.globalAlpha = 1;
                }
            }

            ctx.globalCompositeOperation = 'source-over';

            // --- labels ---
            ctx.font = '11px Jost, sans-serif';
            ctx.textBaseline = 'middle';

            // input feature names + activation bars
            for (i = 0; i < LAYERS[0]; i++) {
                var inNd = nodes[0][i];
                ctx.textAlign = 'right';
                ctx.fillStyle = 'rgba(154,151,163,0.75)';
                ctx.fillText(FEATURES[i], inNd.x - 16, inNd.y - 7);
                var bw = 46;
                ctx.fillStyle = 'rgba(255,255,255,0.10)';
                ctx.fillRect(inNd.x - 16 - bw, inNd.y + 3, bw, 3);
                ctx.fillStyle = AMBER;
                ctx.fillRect(inNd.x - 16 - bw, inNd.y + 3, bw * (prop ? prop.acts[0][i] : 0), 3);
            }

            // output label + verdict
            var out = nodes[LAYERS.length - 1][0];
            ctx.textAlign = 'left';
            ctx.fillStyle = 'rgba(154,151,163,0.85)';
            ctx.font = '12px Jost, sans-serif';
            ctx.fillText('PINK FLOYD?', out.x + 24, out.y - 14);
            if (prop && (prop.instant || now - prop.t0 >= TOTAL)) {
                var yes = prop.song.floyd;
                ctx.font = '600 20px Jost, sans-serif';
                ctx.fillStyle = yes ? YES_COLOR : NO_COLOR;
                ctx.fillText(yes ? 'YES' : 'NO', out.x + 24, out.y + 10);
                if (!prop.verdictShown) {
                    prop.verdictShown = true;
                    showVerdict(prop.song);
                }
            }

            // song being played (bottom right, outside the scrim)
            if (prop) {
                ctx.font = '11px Jost, sans-serif';
                ctx.textAlign = 'right';
                ctx.fillStyle = 'rgba(154,151,163,0.7)';
                ctx.fillText('♫ input: ' + prop.song.name, W - 30, H - 24);
            }

            // hover: inspect a node's activation
            if (finePointer) {
                for (l = 0; l < LAYERS.length; l++) {
                    for (i = 0; i < LAYERS[l]; i++) {
                        var hn = nodes[l][i];
                        if (Math.hypot(mouse.x - hn.x, mouse.y - hn.y) < 22) {
                            ctx.strokeStyle = 'rgba(217,142,63,0.9)';
                            ctx.lineWidth = 1.2;
                            ctx.beginPath();
                            ctx.arc(hn.x, hn.y, 12, 0, Math.PI * 2);
                            ctx.stroke();
                            ctx.fillStyle = AMBER;
                            ctx.textAlign = 'center';
                            ctx.fillText('a = ' + hn.a.toFixed(2), hn.x, hn.y - 22);
                        }
                    }
                }
            }
        }

        if (reduced) {
            // settle activations, then render one static frame
            for (var k = 0; k < 60; k++) draw(performance.now());
        } else {
            (function loop() {
                draw(performance.now());
                requestAnimationFrame(loop);
            })();
        }
    }

    /* ------------------------------------------------------------
       3D tilt on cards
       ------------------------------------------------------------ */
    if (finePointer && !reduced) {
        var cards = document.querySelectorAll(
            '.experience-item, .publication-item, .article-item, .education-item, .stat-chip');
        cards.forEach(function (card) {
            card.addEventListener('mousemove', function (e) {
                var r = card.getBoundingClientRect();
                var px = (e.clientX - r.left) / r.width - 0.5;
                var py = (e.clientY - r.top) / r.height - 0.5;
                card.style.transform =
                    'perspective(700px) rotateX(' + (py * -5).toFixed(2) + 'deg)' +
                    ' rotateY(' + (px * 6).toFixed(2) + 'deg) translateY(-2px)';
            });
            card.addEventListener('mouseleave', function () {
                card.style.transform = '';
            });
        });
    }

    /* ------------------------------------------------------------
       Count-up stats
       ------------------------------------------------------------ */
    var nums = document.querySelectorAll('.stat-num');
    if (nums.length) {
        var statIO = new IntersectionObserver(function (entries) {
            entries.forEach(function (en) {
                if (!en.isIntersecting) return;
                statIO.unobserve(en.target);
                var el = en.target;
                var target = parseFloat(el.dataset.target || '0');
                var pre = el.dataset.prefix || '';
                var suf = el.dataset.suffix || '';
                if (reduced) { el.textContent = pre + target + suf; return; }
                var t0 = performance.now(), dur = 1400;
                (function tick(t) {
                    var k = Math.min(1, (t - t0) / dur);
                    var ease = 1 - Math.pow(1 - k, 3);
                    el.textContent = pre + Math.round(target * ease) + suf;
                    if (k < 1) requestAnimationFrame(tick);
                })(t0);
            });
        }, { threshold: 0.4 });
        nums.forEach(function (n) { statIO.observe(n); });
    }

    /* ------------------------------------------------------------
       The heartbeat — a proper ECG trace (P, QRS, T),
       ~62 bpm, like the pulse that opens and closes DSOTM
       ------------------------------------------------------------ */
    var pulse = document.getElementById('pulse-canvas');
    if (pulse) initPulse(pulse);

    function initPulse(cv) {
        var ctx = cv.getContext('2d');
        var dpr = Math.min(devicePixelRatio || 1, 2);

        function size() {
            cv.width = cv.clientWidth * dpr;
            cv.height = cv.clientHeight * dpr;
        }
        size();
        addEventListener('resize', size);

        // One cardiac cycle, x in [0,1): sum of gaussians.
        // P wave, PR segment, QRS complex, ST segment, T wave.
        function gauss(x, amp, mu, sd) {
            var d = (x - mu) / sd;
            return amp * Math.exp(-d * d / 2);
        }
        function ecg(x) {
            return gauss(x, 0.14, 0.16, 0.024)   // P
                + gauss(x, -0.10, 0.355, 0.009)  // Q
                + gauss(x, 1.00, 0.385, 0.011)   // R
                + gauss(x, -0.24, 0.415, 0.011)  // S
                + gauss(x, 0.30, 0.60, 0.042);   // T
        }

        var CYCLE = 58;          // frames per beat (~62 bpm at 60fps)
        var t = 0;
        var vals = [];

        function next() {
            t++;
            var x = (t % CYCLE) / CYCLE;
            var v = ecg(x);
            v += Math.sin(t * 0.013) * 0.015;          // baseline wander
            v += (Math.random() - 0.5) * 0.012;        // muscle noise
            return v;
        }

        function draw() {
            var W = cv.width, H = cv.height;
            var max = Math.max(80, Math.floor(W / (2 * dpr)));
            vals.push(next());
            while (vals.length > max) vals.shift();

            ctx.clearRect(0, 0, W, H);

            // faint monitor grid
            ctx.strokeStyle = 'rgba(255,255,255,0.045)';
            ctx.lineWidth = 1;
            var step = 14 * dpr;
            ctx.beginPath();
            for (var gx = 0; gx < W; gx += step) { ctx.moveTo(gx, 0); ctx.lineTo(gx, H); }
            for (var gy = 0; gy < H; gy += step) { ctx.moveTo(0, gy); ctx.lineTo(W, gy); }
            ctx.stroke();

            // warm trace: rust -> amber -> steel
            var grad = ctx.createLinearGradient(0, 0, W, 0);
            for (var i = 0; i < SPECTRUM.length; i++) {
                grad.addColorStop(i / (SPECTRUM.length - 1), SPECTRUM[i]);
            }
            ctx.strokeStyle = grad;
            ctx.lineWidth = 1.6 * dpr;
            ctx.lineJoin = 'round';
            ctx.beginPath();
            var base = H * 0.72, scale = H * 0.52;
            for (i = 0; i < vals.length; i++) {
                var x = (i / (max - 1)) * W;
                var y = base - vals[i] * scale;
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // bright leading dot, like a monitor sweep
            if (vals.length > 1) {
                var lx = ((vals.length - 1) / (max - 1)) * W;
                var ly = base - vals[vals.length - 1] * scale;
                ctx.fillStyle = '#ffffff';
                ctx.beginPath();
                ctx.arc(lx, ly, 2.2 * dpr, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        if (reduced) {
            for (var i = 0; i < 400; i++) vals.push(next());
            draw();
        } else {
            (function loop() {
                draw();
                requestAnimationFrame(loop);
            })();
        }
    }

})();
