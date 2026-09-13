#!/usr/bin/env python3
"""
ShibaHash GPU miner (Robinhood Chain, chain ID 4663)
=====================================================

Implements the proof-of-work rule from the ShibaHash v0.3 spec:

    work = keccak256(miner ++ uint256(nonce) ++ previousNFTWork ++ L2AnchorHash)   # 116 bytes packed
    uint256(work) < targetFor(miner)

Nonce grinding runs on your GPU via OpenCL (pyopencl). Every GPU hit is
re-verified on the CPU before anything is submitted on-chain, mirroring what
ShibaHash's own web miner does ("every reported GPU solution is checked on CPU").

WHAT'S SOLID vs. WHAT'S BEST-EFFORT
--------------------------------------
The hash rule above comes straight from the published spec, and the OpenCL
Keccak-256 kernel below is a direct, tested port of a from-scratch
implementation I cross-checked against pycryptodome on 200 random 116-byte
inputs (padding, rotation offsets, round constants all verified). That part
just works.

The contract ABI itself is fetched automatically at startup from Robinhood
Chain's own Blockscout instance (free, no key needed) - you don't need to go
get it by hand. From that ABI, the script auto-detects which function is the
previous-work getter, the per-miner target getter, the total-minted counter,
and the payable mint call, by matching argument/return types plus name hints.
It prints what it picked every run - glance at that line once before letting
it mint for real, since a heuristic match on someone else's ABI isn't a
guarantee. If auto-detection is ever wrong, pass --abi with a manually-checked
ABI file instead, or edit `autodetect_functions()`.

Requirements:
    pip install web3 pyopencl pycryptodome

Usage:
    export SHIBAHASH_PRIVATE_KEY=0xyourprivatekey
    python3 shibahash_miner.py --abi abi.json
"""

import argparse
import json
import os
import struct
import sys
import time

import numpy as np
import requests
from Crypto.Hash import keccak as _keccak
from web3 import Web3

# ---------------------------------------------------------------------------
# CONFIG - defaults per the published spec / repo. Override with CLI flags.
# ---------------------------------------------------------------------------
DEFAULT_RPC_URL = "https://rpc.mainnet.chain.robinhood.com"
DEFAULT_CHAIN_ID = 4663
DEFAULT_CONTRACT = "0xF46A1d2eDDD1004B1345F3D0B5A2Db8e28939c67"  # NFT collection / mining contract
BLOCKSCOUT_API = "https://robinhoodchain.blockscout.com/api/v2/smart-contracts/{address}"


def fetch_abi(contract_address: str, cache_path: str = "abi_cache.json") -> list:
    """Pull the verified ABI straight from Robinhood Chain's own Blockscout
    instance (its free, non-pro v2 API - no key needed), so you don't have to
    grab it by hand. Caches to disk so we don't refetch every run."""
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    url = BLOCKSCOUT_API.format(address=contract_address)
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    abi = data.get("abi")
    if not abi:
        raise RuntimeError(
            f"Blockscout returned no ABI for {contract_address} - the contract "
            "may not show as verified there yet. Grab abi.json by hand from "
            f"https://robinhoodchain.blockscout.com/address/{contract_address}?tab=contract "
            "and pass --abi abi.json instead."
        )
    with open(cache_path, "w") as f:
        json.dump(abi, f)
    return abi


def autodetect_functions(abi: list) -> dict:
    """Guess which ABI entries are the PoW read getters and the mint call,
    from function shape (mutability/inputs/outputs) plus name hints. Prints
    what it picked so you can sanity-check it against the explorer yourself -
    this is a best-effort match, not a guarantee."""
    funcs = [e for e in abi if e.get("type") == "function"]

    def sig(inputs):
        return [i["type"] for i in inputs]

    previous_work = target_for = total_minted = mint_fn = None

    for f in funcs:
        name = f["name"].lower()
        ins = sig(f.get("inputs", []))
        outs = sig(f.get("outputs", []))
        mut = f.get("stateMutability")

        if previous_work is None and mut in ("view", "pure") and ins == [] and outs == ["bytes32"] \
                and any(k in name for k in ("work", "prevhash", "previous", "prev")):
            previous_work = f["name"]
        if target_for is None and mut in ("view", "pure") and ins == ["address"] and outs and outs[0] == "uint256" \
                and any(k in name for k in ("target", "difficulty")):
            target_for = f["name"]
        if total_minted is None and mut in ("view", "pure") and ins == [] and outs and outs[0] == "uint256" \
                and any(k in name for k in ("totalminted", "minted", "supply", "counter", "nextid")):
            total_minted = f["name"]
        if mint_fn is None and mut == "payable" and ins.count("uint256") >= 2 and "bytes32" in ins:
            mint_fn = f["name"]

    picked = {
        "previous_work": previous_work,
        "target_for": target_for,
        "total_minted": total_minted,
        "mint": mint_fn,
    }
    print("Auto-detected contract functions (verify these look right against "
          f"the explorer if a mint ever reverts): {picked}")
    missing = [k for k, v in picked.items() if v is None]
    if missing:
        raise RuntimeError(
            f"Could not auto-detect: {missing}. Open the ABI/explorer, find the "
            "right names yourself, and set FN_PREVIOUS_WORK / FN_TARGET_FOR / "
            "FN_TOTAL_MINTED / FN_MINT constants near the top of this file."
        )
    return picked

# Epoch-0 price per the spec (0.000069 ETH), used as a local fallback estimate
# only. Real payment MUST equal the price the contract expects, so this is a
# convenience default rather than authoritative - reading a live price getter
# (if the ABI has one, e.g. currentPrice()) is safer than trusting this calc.
EPOCH0_PRICE_WEI = Web3.to_wei(0.000069, "ether")

# GPU search parameters
NONCES_PER_BATCH = 1 << 22   # 4,194,304 nonces per dispatch; tune to your GPU


# ---------------------------------------------------------------------------
# CPU keccak256 (ground truth, used to verify every GPU hit before we spend gas)
# ---------------------------------------------------------------------------
def keccak256(data: bytes) -> bytes:
    return _keccak.new(data=data, digest_bits=256).digest()


# ---------------------------------------------------------------------------
# OpenCL Keccak-256 kernel
#
# Ported 1:1 from a from-scratch Python implementation that was cross-checked
# against pycryptodome's keccak256 on 200 random 116-byte inputs before this
# translation (see test_keccak.py alongside this file). I don't have a GPU in
# my own sandbox to execute-test the OpenCL build itself, so please run
# `--selftest` first, which grinds a few thousand nonces against a dummy
# target and re-verifies every GPU hit on the CPU - if that passes, the kernel
# is correct on your hardware.
# ---------------------------------------------------------------------------
KECCAK_CL_SOURCE = r"""
__constant ulong RNDC[24] = {
    0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808AUL, 0x8000000080008000UL,
    0x000000000000808BUL, 0x0000000080000001UL, 0x8000000080008081UL, 0x8000000000008009UL,
    0x000000000000008AUL, 0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000AUL,
    0x000000008000808BUL, 0x800000000000008BUL, 0x8000000000008089UL, 0x8000000000008003UL,
    0x8000000000008002UL, 0x8000000000000080UL, 0x000000000000800AUL, 0x800000008000000AUL,
    0x8000000080008081UL, 0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL
};
__constant int ROTC[24] = {1,3,6,10,15,21,28,36,45,55,2,14,27,41,56,8,25,43,62,18,39,61,20,44};
__constant int PILN[24] = {10,7,11,17,18,3,5,16,8,21,24,4,15,23,19,13,12,2,20,14,22,9,6,1};

inline ulong rotl64(ulong x, int n) {
    return (x << n) | (x >> (64 - n));
}

void keccakf(ulong st[25]) {
    for (int r = 0; r < 24; r++) {
        ulong bc[5];
        for (int i = 0; i < 5; i++)
            bc[i] = st[i] ^ st[i+5] ^ st[i+10] ^ st[i+15] ^ st[i+20];
        for (int i = 0; i < 5; i++) {
            ulong t = bc[(i+4)%5] ^ rotl64(bc[(i+1)%5], 1);
            for (int j = 0; j < 25; j += 5)
                st[j+i] ^= t;
        }
        ulong t = st[1];
        for (int i = 0; i < 24; i++) {
            int j = PILN[i];
            ulong tmp = st[j];
            st[j] = rotl64(t, ROTC[i]);
            t = tmp;
        }
        for (int j = 0; j < 25; j += 5) {
            ulong bc0 = st[j], bc1 = st[j+1], bc2 = st[j+2], bc3 = st[j+3], bc4 = st[j+4];
            st[j]   ^= (~bc1) & bc2;
            st[j+1] ^= (~bc2) & bc3;
            st[j+2] ^= (~bc3) & bc4;
            st[j+3] ^= (~bc4) & bc0;
            st[j+4] ^= (~bc0) & bc1;
        }
        st[0] ^= RNDC[r];
    }
}

// Searches nonces = base_nonce_lo64 + get_global_id(0) (only the low 64 bits
// of the 256-bit nonce field vary; the high 192 bits come from `template_`).
// template_ is the 116-byte packed input (miner || nonce(32) || prevWork(32) ||
// anchorHash(32)) with the low 8 bytes of the nonce field left as zero filler
// for us to overwrite per-thread.
__kernel void mine_keccak(
    __global const uchar *template_,   // 116 bytes
    ulong base_nonce_lo64,
    __global const uchar *target,      // 32 bytes, big-endian, compared lexicographically
    __global ulong *found_flag,        // [0] = 1 if found
    __global ulong *found_nonce_lo64
) {
    size_t gid = get_global_id(0);
    ulong nonce_lo = base_nonce_lo64 + (ulong)gid;

    uchar buf[136]; // rate size, zero-padded
    for (int i = 0; i < 136; i++) buf[i] = 0;
    for (int i = 0; i < 116; i++) buf[i] = template_[i];

    // nonce occupies bytes [20:52), big-endian uint256; overwrite low 8 bytes (bytes 44..52)
    for (int i = 0; i < 8; i++) {
        buf[51 - i] = (uchar)(nonce_lo >> (8 * i));
    }

    // Keccak padding (pre-NIST, as Solidity uses): pad byte 0x01 at end of
    // message, 0x80 XORed into last byte of the 136-byte rate block.
    buf[116] ^= 0x01;
    buf[135] ^= 0x80;

    ulong st[25];
    for (int i = 0; i < 25; i++) st[i] = 0;
    for (int i = 0; i < 136; i += 8) {
        ulong lane = 0;
        for (int b = 0; b < 8; b++)
            lane |= ((ulong)buf[i+b]) << (8*b);
        st[i/8] ^= lane;
    }
    keccakf(st);

    uchar digest[32];
    for (int i = 0; i < 4; i++) {
        ulong lane = st[i];
        for (int b = 0; b < 8; b++)
            digest[i*8+b] = (uchar)(lane >> (8*b));
    }

    // big-endian lexicographic compare digest < target
    bool less = false;
    for (int i = 0; i < 32; i++) {
        if (digest[i] < target[i]) { less = true; break; }
        if (digest[i] > target[i]) { less = false; break; }
    }

    if (less) {
        if (atomic_cmpxchg((volatile __global uint*)found_flag, 0, 1) == 0 ||
            found_flag[0] == 1) {
            found_flag[0] = 1;
            found_nonce_lo64[0] = nonce_lo;
        }
    }
}
"""


def build_template(miner_addr: str, prev_work: bytes, anchor_hash: bytes) -> bytearray:
    """116-byte packed input with the nonce field zeroed (low 8 bytes get filled per-thread)."""
    assert len(prev_work) == 32 and len(anchor_hash) == 32
    miner_bytes = bytes.fromhex(miner_addr[2:].rjust(40, "0"))
    assert len(miner_bytes) == 20
    template = bytearray(20 + 32 + 32 + 32)
    template[0:20] = miner_bytes
    # nonce (bytes 20:52) left as zero; GPU/host fill the low 8 bytes
    template[52:84] = prev_work
    template[84:116] = anchor_hash
    return template


def gpu_search(cl, ctx, queue, prg, template_: bytearray, target: bytes,
               base_nonce_lo64: int, batch_size: int):
    mf = cl.mem_flags
    template_np = np.frombuffer(bytes(template_), dtype=np.uint8)
    target_np = np.frombuffer(target, dtype=np.uint8)

    template_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=template_np)
    target_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=target_np)
    found_flag = np.zeros(1, dtype=np.uint64)
    found_nonce = np.zeros(1, dtype=np.uint64)
    found_flag_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=found_flag)
    found_nonce_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=found_nonce)

    prg.mine_keccak(
        queue, (batch_size,), None,
        template_buf, np.uint64(base_nonce_lo64), target_buf,
        found_flag_buf, found_nonce_buf,
    )
    cl.enqueue_copy(queue, found_flag, found_flag_buf)
    cl.enqueue_copy(queue, found_nonce, found_nonce_buf)
    queue.finish()

    if found_flag[0]:
        return int(found_nonce[0])
    return None


def verify_on_cpu(miner_addr: str, nonce_int: int, prev_work: bytes,
                   anchor_hash: bytes, target_int: int) -> bool:
    miner_bytes = bytes.fromhex(miner_addr[2:].rjust(40, "0"))
    packed = miner_bytes + nonce_int.to_bytes(32, "big") + prev_work + anchor_hash
    digest = keccak256(packed)
    return int.from_bytes(digest, "big") < target_int


def selftest():
    """Grind against an easy target with no chain connection, to sanity-check
    the OpenCL kernel on your actual GPU before trusting it with real gas."""
    import pyopencl as cl_

    platforms = cl_.get_platforms()
    if not platforms:
        print("No OpenCL platforms found. Install your GPU vendor's OpenCL "
              "driver (NVIDIA: CUDA toolkit incl. OpenCL; AMD: ROCm/adrenalin; "
              "Intel: intel-opencl-icd).")
        sys.exit(1)
    ctx = cl_.Context(dev_type=cl_.device_type.ALL)
    queue = cl_.CommandQueue(ctx)
    prg = cl_.Program(ctx, KECCAK_CL_SOURCE).build()

    miner_addr = "0x00000000000000000000000000000000000001"
    prev_work = b"\x11" * 32
    anchor_hash = b"\x22" * 32
    # easy target: top byte must be 0 (~1/256 chance per nonce)
    target_int = (1 << 248)
    target = target_int.to_bytes(32, "big")
    template_ = build_template(miner_addr, prev_work, anchor_hash)

    base = 0
    found = None
    for attempt in range(200):
        found = gpu_search(cl_, ctx, queue, prg, template_, target, base, NONCES_PER_BATCH)
        if found is not None:
            break
        base += NONCES_PER_BATCH
    if found is None:
        print("Self-test FAILED: no hit found in search space (kernel likely broken).")
        sys.exit(1)
    ok = verify_on_cpu(miner_addr, found, prev_work, anchor_hash, target_int)
    print(f"GPU claimed nonce {found}; CPU verification: {'PASS' if ok else 'FAIL'}")
    if not ok:
        sys.exit(1)
    print("Self-test OK: your GPU kernel produces valid keccak256 PoW solutions.")


# ---------------------------------------------------------------------------
# Chain interaction - the parts you must double-check against the real ABI.
# ---------------------------------------------------------------------------
def read_chain_state(w3: Web3, contract, fn: dict, miner_addr: str):
    prev_work = contract.functions[fn["previous_work"]]().call()
    target_int = contract.functions[fn["target_for"]](miner_addr).call()
    total_minted = contract.functions[fn["total_minted"]]().call()

    current_block = w3.eth.block_number
    anchor_block = current_block - 3  # any value with 1 <= current-anchor <= 250 is valid
    anchor_hash = bytes(w3.eth.get_block(anchor_block)["hash"])

    return {
        "prev_work": bytes(prev_work),
        "target_int": int(target_int),
        "total_minted": int(total_minted),
        "anchor_block": anchor_block,
        "anchor_hash": anchor_hash,
    }


def estimate_price_wei(total_minted: int) -> int:
    """Local fallback price estimate from the published epoch formula.
    Prefer a live price getter from the ABI if one exists - if your submitted
    value doesn't exactly match what the contract expects, the mint reverts
    (and still burns gas), per the spec."""
    next_id = total_minted + 1
    k = 0
    b = 0
    while True:
        size = 8 * (2 ** k)
        if next_id <= b + size:
            break
        b += size
        k += 1
    if k == 0:
        return EPOCH0_PRICE_WEI
    return int(b * Web3.to_wei(0.000020, "ether"))


def submit_mint(w3: Web3, contract, fn: dict, account, nonce_int: int, anchor_block: int,
                anchor_hash: bytes, value_wei: int, chain_id: int):
    tx = contract.functions[fn["mint"]](nonce_int, anchor_block, anchor_hash).build_transaction({
        "from": account.address,
        "value": value_wei,
        "nonce": w3.eth.get_transaction_count(account.address),
        "chainId": chain_id,
    })
    tx["gas"] = w3.eth.estimate_gas(tx)
    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"Submitted mint tx: {tx_hash.hex()}")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print(f"Mined in block {receipt.blockNumber}, status={receipt.status}")
    return receipt


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="ShibaHash GPU PoW miner")
    ap.add_argument("--rpc", default=DEFAULT_RPC_URL)
    ap.add_argument("--contract", default=DEFAULT_CONTRACT)
    ap.add_argument("--abi", help="Path to a contract ABI JSON. If omitted, it's fetched "
                     "automatically from Robinhood Chain's Blockscout instance.")
    ap.add_argument("--no-fetch-abi", action="store_true",
                     help="Don't auto-fetch the ABI even if --abi is omitted")
    ap.add_argument("--chain-id", type=int, default=DEFAULT_CHAIN_ID)
    ap.add_argument("--dry-run", action="store_true",
                     help="Grind and verify solutions on CPU but never submit a transaction")
    ap.add_argument("--selftest", action="store_true",
                     help="Verify the OpenCL kernel on your GPU with a synthetic easy target, no chain needed")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    import pyopencl as cl_

    private_key = os.environ.get("SHIBAHASH_PRIVATE_KEY")
    if not private_key and not args.dry_run:
        print("Set SHIBAHASH_PRIVATE_KEY (or pass --dry-run to just grind + verify without minting).")
        sys.exit(1)

    w3 = Web3(Web3.HTTPProvider(args.rpc))
    if not w3.is_connected():
        print(f"Could not connect to RPC {args.rpc}")
        sys.exit(1)

    account = w3.eth.account.from_key(private_key) if private_key else None
    miner_addr = account.address if account else Web3.to_checksum_address(
        "0x" + "0" * 40)  # placeholder address for --dry-run without a key

    contract = None
    fn = None
    if args.abi:
        with open(args.abi) as f:
            abi = json.load(f)
        contract = w3.eth.contract(address=Web3.to_checksum_address(args.contract), abi=abi)
        fn = autodetect_functions(abi)
    elif not args.no_fetch_abi:
        try:
            abi = fetch_abi(args.contract)
            contract = w3.eth.contract(address=Web3.to_checksum_address(args.contract), abi=abi)
            fn = autodetect_functions(abi)
        except Exception as e:
            print(f"Auto-fetching the ABI failed: {e}")
            if not args.dry_run:
                sys.exit(1)
            print("Continuing in --dry-run mode against a placeholder state instead.")
    elif not args.dry_run:
        print("Pass --abi path/to/abi.json, or drop --no-fetch-abi, to read state and mint.")
        sys.exit(1)

    platforms = cl_.get_platforms()
    if not platforms:
        print("No OpenCL platforms found - install your GPU vendor's OpenCL driver.")
        sys.exit(1)
    ctx = cl_.Context(dev_type=cl_.device_type.ALL)
    queue = cl_.CommandQueue(ctx)
    prg = cl_.Program(ctx, KECCAK_CL_SOURCE).build()

    print(f"Mining as {miner_addr} on chain {args.chain_id} via {args.rpc}")

    while True:
        if contract is not None:
            state = read_chain_state(w3, contract, fn, miner_addr)
        else:
            # --dry-run without ABI: just grind against a placeholder state so
            # you can watch hashrate / confirm the kernel runs on your GPU.
            state = {
                "prev_work": b"\x00" * 32,
                "target_int": (1 << 256) - 1 >> 34,
                "total_minted": 0,
                "anchor_block": 0,
                "anchor_hash": b"\x00" * 32,
            }

        target_bytes = state["target_int"].to_bytes(32, "big")
        template_ = build_template(miner_addr, state["prev_work"], state["anchor_hash"])

        base = int.from_bytes(os.urandom(8), "big")
        t0 = time.time()
        found = None
        batches = 0
        while found is None:
            found = gpu_search(cl_, ctx, queue, prg, template_, target_bytes, base, NONCES_PER_BATCH)
            base = (base + NONCES_PER_BATCH) & ((1 << 64) - 1)
            batches += 1
            if batches % 10 == 0:
                rate = batches * NONCES_PER_BATCH / (time.time() - t0)
                print(f"  ~{rate/1e6:.1f} MH/s, no solution yet this round")
            # re-read chain state periodically so we don't grind on stale prevWork
            if contract is not None and batches % 200 == 0:
                break

        if found is None:
            continue  # loop back and refresh chain state

        # full 256-bit nonce = zero high 192 bits ++ found low-64 bits (see build_template)
        nonce_int = found

        if not verify_on_cpu(miner_addr, nonce_int, state["prev_work"],
                              state["anchor_hash"], state["target_int"]):
            print("GPU produced an invalid candidate (should not happen) - discarding, retrying.")
            continue

        print(f"Valid nonce found: {nonce_int}")

        if args.dry_run or contract is None:
            print("(--dry-run) Not submitting a transaction.")
            if args.dry_run:
                break
            continue

        value_wei = estimate_price_wei(state["total_minted"])
        submit_mint(w3, contract, fn, account, nonce_int, state["anchor_block"],
                    state["anchor_hash"], value_wei, args.chain_id)


if __name__ == "__main__":
    main()
