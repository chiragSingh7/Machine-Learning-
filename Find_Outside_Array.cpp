#include <bits/stdc++.h>
using namespace std;

#define endl '\n'
#define pb push_back
#define mp make_pair
#define F first
#define S second
#define All(x) (x).begin(), (x).end()
#define input(arr) for (auto &x: arr) cin >> x;
#define output(arr, ...) for (auto &x: arr) cout << x << __VA_ARGS__;
#define sz(a) ((int)(a.size()))
#define ll long long
#define int long long
#define tmax(a, b, c) max(a, max(b, c))
#define tmin(a, b, c) min(a, min(b, c))
#define tcon(cond) ((cond) ? "YES" : "NO")
typedef vector<int> vi;
typedef vector<bool> vb;
typedef vector<string> vs;
typedef vector<ll> vll;
typedef vector<vector<int>> vvi;
typedef vector<vector<bool>> vvb;
typedef vector<vector<ll>> vvll;
typedef vector<pair<int, int>> vpii;
typedef vector<pair<ll, ll>> vpll;
typedef pair<int, int> pii;
typedef pair<ll, ll> pll;
typedef unordered_map<int, int> umii;
typedef unordered_map<char, int> umci;
typedef list<int> li;
const ll MOD = 1e9 + 7;
const long double eps = 1e-9;
const int INF = 1e18;
const int MAXN = 1e5 + 5;

int binpow(int a, int b, int mod) {
    int res = 1;
    while (b > 0) {
        if (b & 1) {
            res = (res * a) % mod;
        }
        a = (a * a) % mod;
        b >>= 1;
    }
    return res;
}
int gcd(int a, int b) {
    return b == 0 ? a : gcd(b, a % b);
}
int lcm(int a, int b) {
    return a / gcd(a, b) * b;
}
int modinv(int a, int m) {
    return binpow(a, m - 2, m);
}
int modadd(int a, int b, int m) {
    return (a + b) % m;
}


void Problem() {
    int n;
    cin >> n;
    vi arr(n);
    input(arr);
    mpii mp;
    for(int i = 0;i<n;i++){
        mp[arr[i]]++;
    }
    for(int i = 0;i<n;i++){
        for(int j = i+1;j<n;j++){
            if(mp.find(arr[i] + arr[j]) == mp.end()){
                cout << arr[i] << " " << arr[j] << endl;
                return;
            }
        }
    }
    cout << -1 << endl;
}

int32_t main() {
    int t;
    t = 1;
    cin >> t;

    while (t--) {
        Problem();
    }

    return 0;
}