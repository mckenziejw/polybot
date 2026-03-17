"""
replay.py — Market replay viewer for trained MaskablePPO models.

Usage:
    python replay.py

Then open http://localhost:5006 in your browser.

Optional args:
    python replay.py --train-dir data/telonex_100ms
                     --test-dir  data/telonex_100ms_test
                     --model     models/ppo_v1_final.zip
                     --normalizer models/ppo_v1_vecnormalize.pkl
                     --port      5006
"""

import argparse
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd

import holoviews as hv
import panel as pn
from holoviews import opts

hv.extension('bokeh')
pn.extension()

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--train-dir',   default='data/telonex_100ms')
parser.add_argument('--test-dir',    default='data/telonex_100ms_test')
parser.add_argument('--model',       default='models/checkpoints/ppo_v1_2299885_steps.zip')
parser.add_argument('--normalizer',  default='models/checkpoints/vecnormalize.pkl')
parser.add_argument('--btc-path',    default='data/btc_quotes/btcusdt_quotes.parquet')
parser.add_argument('--resolution',  default='data/resolutions.json')
parser.add_argument('--port',        type=int, default=5006)
args = parser.parse_args()

TRAIN_DIR       = Path(args.train_dir)
TEST_DIR        = Path(args.test_dir)
MODEL_PATH      = Path(args.model)
NORMALIZER_PATH = Path(args.normalizer)
BTC_PATH        = Path(args.btc_path)
RESOLUTION_PATH = Path(args.resolution)

for p in [MODEL_PATH, NORMALIZER_PATH, BTC_PATH, RESOLUTION_PATH]:
    if not p.exists():
        raise FileNotFoundError(f'Missing: {p}')

# ---------------------------------------------------------------------------
# Load model (once, at startup)
# ---------------------------------------------------------------------------

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from polymarket_env import PolymarketEnv

print(f'Loading model from {MODEL_PATH}...')
model = MaskablePPO.load(MODEL_PATH, device='cpu')
print('Model loaded.')

# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

ACTION_NAMES = {
    0: 'Hold',
    1: 'Buy Yes (S)', 2: 'Buy Yes (L)',
    3: 'Buy No (S)',  4: 'Buy No (L)',
    5: 'Sell',
}

def run_episode(slug, book_dir):
    def _make():
        return PolymarketEnv(
            book_dir=book_dir,
            btc_path=BTC_PATH,
            resolution_path=str(RESOLUTION_PATH),
            market_slugs=[slug],
            deterministic=True,
        )

    raw_env = DummyVecEnv([_make])
    env = VecNormalize.load(str(NORMALIZER_PATH), raw_env)
    env.training    = False
    env.norm_reward = False

    obs = env.reset()
    inner = env.venv.envs[0]

    records = {
        'timestamps': [], 'yes_mid': [], 'no_mid': [],
        'actions': [], 'yes_position': [], 'no_position': [],
        'bankroll': [], 'realized_pnl': [], 'step_rewards': [],
        'yes_book': [], 'no_book': [], 'fees_paid': [],
    }

    terminal_reward = 0.0

    while True:
        masks  = np.array([inner.action_masks()])
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)

        idx     = min(inner._step_idx, len(inner._timestamps) - 1)
        ts      = inner._timestamps[idx]
        yes_row = inner._yes_df.iloc[idx]
        no_row  = inner._no_df.iloc[idx]

        yes_mid = float(yes_row.get('mid_price', 0.5))
        no_mid  = float(no_row.get('mid_price',  0.5))

        N = 5
        yes_book = (
            [float(yes_row.get(f'bid_price_{i}', 0)) for i in range(1, N+1)],
            [float(yes_row.get(f'bid_size_{i}',  0)) for i in range(1, N+1)],
            [float(yes_row.get(f'ask_price_{i}', 0)) for i in range(1, N+1)],
            [float(yes_row.get(f'ask_size_{i}',  0)) for i in range(1, N+1)],
        )
        no_book = (
            [float(no_row.get(f'bid_price_{i}', 0)) for i in range(1, N+1)],
            [float(no_row.get(f'bid_size_{i}',  0)) for i in range(1, N+1)],
            [float(no_row.get(f'ask_price_{i}', 0)) for i in range(1, N+1)],
            [float(no_row.get(f'ask_size_{i}',  0)) for i in range(1, N+1)],
        )

        records['timestamps'].append(int(ts))
        records['yes_mid'].append(yes_mid)
        records['no_mid'].append(no_mid)
        records['actions'].append(int(action[0]))
        records['yes_position'].append(inner._yes_position)
        records['no_position'].append(inner._no_position)
        records['bankroll'].append(inner._bankroll)
        records['realized_pnl'].append(inner._realized_pnl)
        records['yes_book'].append(yes_book)
        records['no_book'].append(no_book)
        records['fees_paid'].append(inner._fees_paid)

        obs, reward, done, info = env.step(action)
        records['step_rewards'].append(float(reward[0]))

        if done[0]:
            terminal_reward = float(reward[0])
            break

    env.close()

    for k in ['timestamps','yes_mid','no_mid','actions',
               'yes_position','no_position','bankroll',
               'realized_pnl','step_rewards','fees_paid']:
        records[k] = np.array(records[k])

    records['elapsed_sec']     = (records['timestamps'] - records['timestamps'][0]) / 1000.0
    records['terminal_reward'] = terminal_reward
    records['yes_resolved']    = inner._yes_resolved
    records['no_resolved']     = inner._no_resolved
    records['slug']            = slug

    return records

# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def make_price_chart(records):
    t        = records['elapsed_sec']
    ymd      = records['yes_mid']
    nmd      = records['no_mid']
    acts     = records['actions']
    yes_pos  = records['yes_position']
    no_pos   = records['no_position']
    slug     = records['slug']
    yes_res  = records['yes_resolved']
    terminal = records['terminal_reward']
    winner   = 'Yes (Up)' if yes_res == 1.0 else 'No (Down)'

    yes_curve = hv.Curve((t, ymd), kdims='Time (s)', vdims='Price', label='Yes'
                         ).opts(color='steelblue', line_width=1.5)
    no_curve  = hv.Curve((t, nmd), kdims='Time (s)', vdims='Price', label='No'
                         ).opts(color='salmon', line_width=1.5)
    mid_line  = hv.HLine(0.5).opts(color='white', line_dash='dashed', line_width=1, alpha=0.4)

    # Position shading
    spans = []
    in_pos, pos_start = None, None
    for i, (yp, np_) in enumerate(zip(yes_pos, no_pos)):
        state = 'yes' if yp > 0 else ('no' if np_ > 0 else None)
        if state != in_pos:
            if in_pos is not None:
                spans.append((pos_start, t[i], in_pos))
            in_pos, pos_start = state, t[i]
    if in_pos is not None:
        spans.append((pos_start, t[-1], in_pos))

    chart = yes_curve * no_curve * mid_line
    for s, e, side in spans:
        chart = chart * hv.VSpan(s, e).opts(
            color='green' if side == 'yes' else 'red', alpha=0.08)

    # Trade markers — build DataFrames so hover tooltips have rich context
    def _marker_df(indices, action_col):
        rows = []
        for i in indices:
            a    = acts[i]
            name = ACTION_NAMES[a]
            price = ymd[i] if a in (1, 2) else (nmd[i] if a in (3, 4) else
                    (ymd[i] if yes_pos[i] > 0 else nmd[i]))
            rows.append({
                'time_s':    round(float(t[i]), 1),
                'price':     round(price, 4),
                'action':    name,
                'yes_mid':   round(float(ymd[i]), 4),
                'no_mid':    round(float(nmd[i]), 4),
                'yes_pos':   round(float(yes_pos[i]), 2),
                'no_pos':    round(float(no_pos[i]), 2),
                'bankroll':  round(float(records['bankroll'][i]), 2),
            })
        return pd.DataFrame(rows) if rows else None

    buy_yes_idx = [i for i, a in enumerate(acts) if a in (1, 2)]
    buy_no_idx  = [i for i, a in enumerate(acts) if a in (3, 4)]
    sell_idx    = [i for i, a in enumerate(acts) if a == 5]

    tooltips = [
        ('Time',     '@time_s s'),
        ('Action',   '@action'),
        ('Price',    '@price{0.0000}'),
        ('Yes mid',  '@yes_mid{0.0000}'),
        ('No mid',   '@no_mid{0.0000}'),
        ('Yes pos',  '$@yes_pos{0.00}'),
        ('No pos',   '$@no_pos{0.00}'),
        ('Bankroll', '$@bankroll{0.00}'),
    ]
    from bokeh.models import HoverTool

    for df, label, color, marker in [
        (_marker_df(buy_yes_idx, 'buy_yes'), '▲ Buy Yes (S=small, L=large)', 'limegreen', 'triangle'),
        (_marker_df(buy_no_idx,  'buy_no'),  '▼ Buy No (S=small, L=large)',  'tomato',    'inverted_triangle'),
        (_marker_df(sell_idx,    'sell'),     '■ Sell (close position)',       'gold',      'square'),
    ]:
        if df is None or df.empty:
            continue
        pts = hv.Points(
            df, kdims=['time_s', 'price'],
            vdims=['action', 'yes_mid', 'no_mid', 'yes_pos', 'no_pos', 'bankroll'],
            label=label,
        ).opts(
            color=color, size=9, marker=marker, line_color='white', line_width=0.5,
            tools=[HoverTool(tooltips=tooltips)],
        )
        chart = chart * pts

    # Legend: add invisible dummy curves so Bokeh renders a clean legend box
    # with plain text labels (the Points labels above are verbose for the legend)
    legend_note = hv.Text(
        0.5, -0.015,
        '▲ Buy Yes   ▼ Buy No   ■ Sell   Green bg = holding Yes   Red bg = holding No',
        fontsize=9,
    ).opts(color='#aaaaaa', text_align='center')
    chart = chart * legend_note

    return chart.opts(opts.Overlay(
        title=f'{slug}  |  Winner: {winner}  |  PnL: ${terminal:+.2f}',
        width=900, height=370, bgcolor='#1a1a2e',
        legend_position='top_left', toolbar='above',
        ylim=(-0.04, 1.02),
    ))


def make_bankroll_chart(records):
    t   = records['elapsed_sec']
    br  = records['bankroll']
    pnl = records['realized_pnl']

    return (
        hv.Curve((t, br),  kdims='Time (s)', vdims='$', label='Bankroll'
                 ).opts(color='cyan', line_width=1.5) *
        hv.Curve((t, pnl), kdims='Time (s)', vdims='$', label='Realized PnL'
                 ).opts(color='yellow', line_width=1.5, line_dash='dashed') *
        hv.HLine(0).opts(color='white', line_dash='dashed', line_width=1, alpha=0.3)
    ).opts(opts.Overlay(
        title='Bankroll & Realized PnL',
        width=900, height=200, bgcolor='#1a1a2e',
        legend_position='top_left', toolbar='above', tools=['hover'],
    ))


def make_book_chart(records, step):
    step = min(step, len(records['yes_book']) - 1)
    t    = records['elapsed_sec'][step]

    def cumulative_depth(bid_p, bid_s, ask_p, ask_s):
        bids = sorted([(p, s) for p, s in zip(bid_p, bid_s) if p > 0 and s > 0], reverse=True)
        asks = sorted([(p, s) for p, s in zip(ask_p, ask_s) if p > 0 and s > 0])
        cum_bid, cum_ask = [], []
        c = 0
        for p, s in bids:
            c += s
            cum_bid.append((p, c))
        c = 0
        for p, s in asks:
            c += s
            cum_ask.append((p, c))
        return cum_bid, cum_ask

    yb = records['yes_book'][step]
    nb = records['no_book'][step]
    y_bids, y_asks = cumulative_depth(*yb)
    n_bids, n_asks = cumulative_depth(*nb)

    elements = []
    if y_bids:
        xs, ys = zip(*y_bids)
        elements.append(hv.Area((xs, ys), kdims='Price', vdims='Size', label='Yes Bid'
                                ).opts(color='steelblue', alpha=0.6))
    if y_asks:
        xs, ys = zip(*y_asks)
        elements.append(hv.Area((xs, ys), kdims='Price', vdims='Size', label='Yes Ask'
                                ).opts(color='steelblue', alpha=0.25))
    if n_bids:
        xs, ys = zip(*n_bids)
        elements.append(hv.Area((xs, ys), kdims='Price', vdims='Size', label='No Bid'
                                ).opts(color='salmon', alpha=0.6))
    if n_asks:
        xs, ys = zip(*n_asks)
        elements.append(hv.Area((xs, ys), kdims='Price', vdims='Size', label='No Ask'
                                ).opts(color='salmon', alpha=0.25))

    if not elements:
        return hv.Curve([]).opts(width=900, height=250)

    chart = elements[0]
    for e in elements[1:]:
        chart = chart * e

    return chart.opts(opts.Overlay(
        title=f'Order Book @ t={t:.1f}s  (step {step})',
        width=900, height=250, bgcolor='#1a1a2e',
        legend_position='top_right', toolbar='above',
        xlim=(0, 1), tools=['hover'],
    ))

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

train_slugs = sorted([f.stem for f in TRAIN_DIR.glob('*.parquet')]) if TRAIN_DIR.exists() else []
test_slugs  = sorted([f.stem for f in TEST_DIR.glob('*.parquet')])  if TEST_DIR.exists()  else []
print(f'Train markets: {len(train_slugs)}, Test markets: {len(test_slugs)}')

split_select = pn.widgets.RadioButtonGroup(
    name='Dataset', options=['Train', 'Test'], value='Train', button_type='default')
slug_select  = pn.widgets.Select(
    name='Market', options=train_slugs, width=500)
random_btn   = pn.widgets.Button(name='Random', button_type='primary', width=100)
run_btn      = pn.widgets.Button(name='Run', button_type='success', width=100)
mode_select  = pn.widgets.RadioButtonGroup(
    name='Mode', options=['Snapshot', 'Scrub'], value='Snapshot', button_type='default')
speed_select = pn.widgets.Select(
    name='Speed', options={'0.5x': 200, '1x': 100, '2x': 50, '5x': 20, '10x': 10},
    value=100, width=100, visible=False)
player       = pn.widgets.Player(
    name='', start=0, end=2999, value=0,
    loop_policy='once', interval=100, width=600, visible=False)
status       = pn.pane.Markdown('*Select a market and click Run.*')
summary_pane = pn.pane.Markdown('')

price_pane    = pn.pane.HoloViews(hv.Curve([]), width=900)
bankroll_pane = pn.pane.HoloViews(hv.Curve([]), width=900)
book_pane     = pn.pane.HoloViews(hv.Curve([]), width=900)

state = {'records': None}

def on_split(event):
    slug_select.options = train_slugs if event.new == 'Train' else test_slugs
    if slug_select.options:
        slug_select.value = slug_select.options[0]

split_select.param.watch(on_split, 'value')

def on_random(event):
    pool = train_slugs if split_select.value == 'Train' else test_slugs
    if pool:
        slug_select.value = np.random.choice(pool)

random_btn.on_click(on_random)

def on_run(event):
    slug = slug_select.value
    if not slug:
        status.object = '**Error:** No market selected.'
        return
    book_dir = TRAIN_DIR if split_select.value == 'Train' else TEST_DIR
    status.object = f'*Running `{slug}`...*'
    run_btn.disabled = True
    try:
        records = run_episode(slug, book_dir)
        state['records'] = records

        n_steps  = len(records['elapsed_sec'])
        n_trades = sum(1 for a in records['actions'] if a != 0)
        n_buys   = sum(1 for a in records['actions'] if a in (1,2,3,4))
        n_sells  = sum(1 for a in records['actions'] if a == 5)
        winner   = 'Yes (Up)' if records['yes_resolved'] == 1.0 else 'No (Down)'

        player.end   = n_steps - 1
        player.value = 0

        total_fees = records['fees_paid'][-1] if len(records['fees_paid']) else 0
        summary_pane.object = (
            f"**Market:** `{slug}`  |  **Winner:** {winner}  |  "
            f"**PnL:** ${records['terminal_reward']:+.2f}  |  "
            f"**Fees paid:** ${total_fees:.2f}  |  "
            f"**Trades:** {n_trades} (buys: {n_buys}, sells: {n_sells})  |  "
            f"**Final bankroll:** ${records['bankroll'][-1]:,.2f}"
        )
        status.object = '*Done.*'
        refresh_charts()
    except Exception as e:
        status.object = f'**Error:** {e}'
        import traceback; traceback.print_exc()
    finally:
        run_btn.disabled = False

run_btn.on_click(on_run)

def on_speed(event):
    player.interval = event.new

speed_select.param.watch(on_speed, 'value')

def on_player(event):
    if state['records'] and mode_select.value == 'Scrub':
        book_pane.object = make_book_chart(state['records'], event.new)

player.param.watch(on_player, 'value')

def on_mode(event):
    is_scrub = event.new == 'Scrub'
    player.visible       = is_scrub
    speed_select.visible = is_scrub
    refresh_charts()

mode_select.param.watch(on_mode, 'value')

def refresh_charts():
    records = state['records']
    if records is None:
        return
    price_pane.object    = make_price_chart(records)
    bankroll_pane.object = make_bankroll_chart(records)
    book_pane.object     = make_book_chart(records, player.value) if mode_select.value == 'Scrub' else hv.Curve([])

app = pn.Column(
    pn.pane.Markdown('# Market Replay Viewer'),
    pn.Row(
        pn.Column(
            split_select,
            slug_select,
            pn.Row(random_btn, run_btn),
            pn.Row(mode_select, speed_select),
            status,
        ),
    ),
    player,
    summary_pane,
    price_pane,
    bankroll_pane,
    book_pane,
)

pn.serve(app, port=args.port, show=True, title='Market Replay')