# CPU geometry model, not shader emulation or attention output validation.
def split(n,cap):
    if n==0:return 1,0
    k=min(cap,(n+255)//256)
    rows=((n+k-1)//k+7)//8*8
    return (n+rows-1)//rows,rows
cases=0
for n in [0,1,7,8,127,128,129,255,256,257,16383,16384,16385,21760,24576,24577,49920,65536]:
 for cap in [64,96]:
  k,rows=split(n,cap)
  for nq in range(2,10):
   for origin in [0,128,256]:
    for tail in [0,128]:
     for causal in [0,1,7,8,127,128,129,max(0,n-1),n,n+1]:
      for qi in range((nq+1)//2*2):
       limit=min(n,max(0,causal+qi)) if qi<nq else 0
       tail_begin=limit-min(limit,tail)
       intervals=[(s*rows,min(limit,(s+1)*rows)) for s in range(k) if min(limit,(s+1)*rows)>s*rows]
       covered=sum(b-a for a,b in intervals)
       body=sum(max(0,min(b,tail_begin)-a) for a,b in intervals)
       tails=sum(max(0,b-max(a,tail_begin)) for a,b in intervals)
       assert covered==limit and body+tails==limit and tails==min(limit,tail)
       assert not intervals or intervals[0][0]==0 and intervals[-1][1]==limit
       assert all(intervals[i][1]==intervals[i+1][0] for i in range(len(intervals)-1))
       if qi>=nq:assert covered==0
       cases+=1
print(f'PASS {cases} causal/phantom/tail window cases; geometry only')
