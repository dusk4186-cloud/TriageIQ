import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Load ChromaDB
CHROMA_PATH = "c:/Users/ASUS/Practice/Jira_MCP/Defect Triage/chroma_db"
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection("jira_defects")

total_docs = collection.count()
print(f"Total documents indexed in ChromaDB: {total_docs}")

# 2. Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. 15 Test Queries with completely rephrased descriptions
test_cases = [
    {
        "target": "SCRUM-6",
        "title": "Inventory oversell during flash sale",
        "query": "We have negative stock numbers on our top-selling items after the holiday promo. Multiple buyers bought the zero-inventory items concurrently."
    },
    {
        "target": "SCRUM-7",
        "title": "Payment webhook silently failing",
        "query": "Stripe shows payments as succeeded, but orders are stuck in pending state and not moving to processing due to webhook verification warnings."
    },
    {
        "target": "SCRUM-8",
        "title": "Tax calculation API timeout",
        "query": "Checkout hangs and eventually fails for customers when computing state sales tax. Looking like a 15-second socket timeout on the tax partner endpoint."
    },
    {
        "target": "SCRUM-9",
        "title": "Loyalty points not deducting on purchase",
        "query": "Buyers are redeeming reward points at checkout, but their rewards balance is not getting debited after the order is placed."
    },
    {
        "target": "SCRUM-12",
        "title": "Database deadlock on order fulfillment",
        "query": "Warehouse workers cannot update order status to shipped. Postgres logs show deadlock detected and ShareLock conflicts between processes."
    },
    {
        "target": "SCRUM-13",
        "title": "Incorrect currency conversion at checkout",
        "query": "European shoppers are being undercharged. The EUR/USD FX rate looks stale in our Redis cache because rates haven't refreshed in 48 hours."
    },
    {
        "target": "SCRUM-14",
        "title": "PDF invoice generation memory leak",
        "query": "Node.js application worker OOM crashing every few hours under heavy traffic. Heap allocation keeps growing when rendering invoice PDFs with Puppeteer."
    },
    {
        "target": "SCRUM-16",
        "title": "Session hijacking via predictable JWT tokens",
        "query": "Pen-test report warns that authorization tokens use weak signing and can be spoofed, resulting in concurrent logins from different IP addresses on a single session."
    },
    {
        "target": "SCRUM-18",
        "title": "Apple Pay token decryption failure",
        "query": "iPhone users cannot finish purchase via Apple Pay. The payment token decryption crashes with invalid PKCS7 padding error on the certificate."
    },
    {
        "target": "SCRUM-19",
        "title": "Promo code stacking logic error",
        "query": "Shoppers are combining multiple discount coupons that should not be stackable, causing cart balances to drop below zero."
    },
    {
        "target": "SCRUM-20",
        "title": "ElasticSearch catalog out of sync",
        "query": "Disabled and out-of-catalog items are still visible in store search results because bulk reindexing failed on a price field float parsing error."
    },
    {
        "target": "SCRUM-23",
        "title": "Duplicate refund processing",
        "query": "Support agents double-clicking the refund button in the CRM issues two payout transactions to the customer for the same charge."
    },
    {
        "target": "SCRUM-25",
        "title": "Redis cache stampede on homepage products",
        "query": "Relational DB hits 100% utilization at the top of every hour because all featured products cache keys expire at the exact same second."
    },
    {
        "target": "SCRUM-30",
        "title": "Gift card double spend race condition",
        "query": "A buyer redeemed the same voucher balance twice across two simultaneous checkout tabs without deducting balance between them."
    },
    {
        "target": "SCRUM-48",
        "title": "Refunds to expired cards fail silently",
        "query": "When refunding an order to a credit card that passed its expiration date, Stripe returns expired_card decline, and the refund gets stuck in limbo without notifying support."
    }
]

top1_hits = 0
top3_hits = 0
top5_hits = 0
reciprocal_ranks = []

results_summary = []

for idx, tc in enumerate(test_cases, 1):
    q_vec = model.encode(tc["query"], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=[q_vec], n_results=5)
    
    retrieved_ids = res["ids"][0]
    retrieved_dists = res["distances"][0]
    
    target = tc["target"]
    rank = None
    sim = None
    if target in retrieved_ids:
        rank = retrieved_ids.index(target) + 1
        dist = retrieved_dists[retrieved_ids.index(target)]
        sim = 1.0 - (dist / 2.0)
        reciprocal_ranks.append(1.0 / rank)
    else:
        reciprocal_ranks.append(0.0)
        
    is_top1 = rank == 1
    is_top3 = rank is not None and rank <= 3
    is_top5 = rank is not None and rank <= 5
    
    if is_top1:
        top1_hits += 1
    if is_top3:
        top3_hits += 1
    if is_top5:
        top5_hits += 1
        
    status = "TOP 1" if is_top1 else (f"TOP {rank}" if is_top3 else (f"Rank {rank}" if rank else "MISSED"))
    
    results_summary.append({
        "num": idx,
        "target": target,
        "title": tc["title"],
        "query": tc["query"],
        "rank": rank,
        "similarity": round(sim, 4) if sim else 0.0,
        "top3_retrieved": retrieved_ids[:3]
    })
    
    print(f"[{idx:02d}/15] Target: {target} | Rank: {rank} | Status: {status} | Top 3: {retrieved_ids[:3]}")

mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
recall_at_1 = (top1_hits / len(test_cases)) * 100
recall_at_3 = (top3_hits / len(test_cases)) * 100
recall_at_5 = (top5_hits / len(test_cases)) * 100

print("\n" + "=" * 60)
print("BENCHMARK EVALUATION RESULTS")
print("=" * 60)
print(f"Total Test Queries: {len(test_cases)}")
print(f"Top-1 Hits (Recall@1): {top1_hits}/{len(test_cases)} ({recall_at_1:.1f}%)")
print(f"Top-3 Hits (Recall@3): {top3_hits}/{len(test_cases)} ({recall_at_3:.1f}%)")
print(f"Top-5 Hits (Recall@5): {top5_hits}/{len(test_cases)} ({recall_at_5:.1f}%)")
print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
print("=" * 60)

with open("c:/Users/ASUS/Practice/Jira_MCP/benchmark_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "total_documents_indexed": total_docs,
        "recall_at_1": recall_at_1,
        "recall_at_3": recall_at_3,
        "recall_at_5": recall_at_5,
        "top3_count": f"{top3_hits} of {len(test_cases)}",
        "mrr": round(mrr, 4),
        "results": results_summary
    }, f, indent=2)

