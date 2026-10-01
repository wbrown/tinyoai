#!/bin/bash
# Build the API binary and its relocatable native runtime. With no SDK argument,
# fetch and build pinned MLX sources in the ignored .build directory.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
case $# in
    0) output="$repo/dist/mlx"; mlx_root= ;;
    1)
        if [[ $1 == --help || $1 == -h ]]; then
            echo "Usage: $0 [OUTPUT_DIRECTORY]"
            echo "       $0 MLX_SDK_ROOT OUTPUT_DIRECTORY"
            echo 'Default: build a cached SDK and package dist/mlx. BUILD_JOBS defaults to 4.'
            exit 0
        fi
        output=$1; mlx_root= ;;
    2) mlx_root=$(cd "$1" && pwd); output=$2 ;;
    *) echo "Usage: $0 [MLX_SDK_ROOT] [OUTPUT_DIRECTORY]" >&2; exit 2 ;;
esac
mlx_revision=1f8e74e3f12f31365464a6867c6579f0e9b29d85
revision=ebc88f10caa1b625e6b581437a8dea6df8a70085
work="$repo/.build/mlx"
sdk="$work/sdk"
jobs=${BUILD_JOBS:-4}
# Check tools before starting downloads or an expensive native build.
if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
    echo 'Build this release on an Apple Silicon Mac.' >&2; exit 1
fi
if [[ ! $jobs =~ ^[1-9][0-9]*$ ]]; then
    echo 'BUILD_JOBS must be a positive integer.' >&2; exit 2
fi
if [[ -e "$output/tinyoai" ]]; then
    echo "Use a new output directory; $output already contains a release." >&2; exit 1
fi
for tool in cmake git go curl; do
    if ! command -v "$tool" >/dev/null; then
        echo "Missing $tool. See docs/mlx-native.md for build prerequisites." >&2; exit 1
    fi
done
# Honor an explicit Xcode selection. Otherwise try the standard application
# when xcode-select points to CommandLineTools. Never change system settings.
if [[ -z ${DEVELOPER_DIR:-} ]] && ! xcrun --find metal >/dev/null 2>&1; then
    if [[ -d /Applications/Xcode.app/Contents/Developer ]]; then
        export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
    fi
fi
if [[ -z $mlx_root ]]; then
    if ! xcrun metal --version >/dev/null 2>&1; then
        echo 'Install Xcode and its Metal Toolchain component. Set DEVELOPER_DIR for a nonstandard Xcode location.' >&2
        exit 1
    fi
    macos_sdk=$(xcrun --sdk macosx --show-sdk-version)
    IFS=. read -r sdk_major sdk_minor sdk_patch <<< "$macos_sdk"
    if (( sdk_major < 26 || (sdk_major == 26 && ${sdk_minor:-0} < 2) )); then
        echo "macOS SDK 26.2 or newer is required; selected SDK is $macos_sdk." >&2; exit 1
    fi
fi
export CGO_ENABLED=1
export GOTOOLCHAIN=${GOTOOLCHAIN:-go1.27.1}
export GOEXPERIMENT=simd
go version

# Fetch a pinned checkout only when absent. Refuse to discard local edits or
# switch an existing checkout to another revision. Completed fetches work offline.
fetch_source() {
    local name=$1 commit=$2 dir="$work/$1"
    if [[ ! -d "$dir/.git" ]]; then
        git init -q "$dir"
        git -C "$dir" remote add origin "https://github.com/ml-explore/$name.git"
    fi
    if [[ -n $(git -C "$dir" status --porcelain) ]]; then
        echo "Local changes in $dir; move that checkout aside before rebuilding." >&2; exit 1
    fi
    if git -C "$dir" rev-parse --verify HEAD >/dev/null 2>&1; then
        if [[ $(git -C "$dir" rev-parse HEAD) != "$commit" ]]; then
            echo "Unexpected revision in $dir; move that checkout aside before rebuilding." >&2; exit 1
        fi
    else
        git -C "$dir" fetch -q --depth 1 origin "$commit"
        git -C "$dir" checkout -q --detach "$commit"
    fi
}

mkdir -p "$work"
if [[ -z $mlx_root ]]; then
    mlx_root="$work/runtime-sdk"
    fetch_source mlx "$mlx_revision"
    echo 'Building MLX 0.32.2 SDK (incremental after the first build).'
    # Use the final runtime path while linking, so repeated installs do not
    # try to rewrite load commands in an already-installed library.
    cmake -S "$work/mlx" -B "$work/runtime-build" \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=26.2 \
        -DCMAKE_INSTALL_PREFIX="$mlx_root" -DCMAKE_INSTALL_LIBDIR=lib \
        -DCMAKE_INSTALL_RPATH=@loader_path -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
        -DFETCHCONTENT_UPDATES_DISCONNECTED=ON \
        -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF \
        -DMLX_BUILD_BENCHMARKS=OFF -DMLX_BUILD_PYTHON_BINDINGS=OFF \
        -DMLX_BUILD_GGUF=OFF -DMLX_METAL_JIT=OFF
    cmake --build "$work/runtime-build" --parallel "$jobs"
    cmake --install "$work/runtime-build"
    mkdir -p "$mlx_root/share/licenses"
    cp "$work/mlx/LICENSE" "$mlx_root/share/licenses/MLX.txt"
    cp "$work/runtime-build/_deps/json-src/LICENSE.MIT" "$mlx_root/share/licenses/nlohmann-json.txt"
    cp "$work/runtime-build/_deps/fmt-src/LICENSE" "$mlx_root/share/licenses/fmt.txt"
fi
for file in lib/libmlx.dylib lib/libjaccl.dylib lib/mlx.metallib include/mlx/mlx.h include/metal_cpp/LICENSE.txt share/cmake/MLX/MLXConfigVersion.cmake; do
    if [[ ! -f "$mlx_root/$file" ]]; then
        echo "Incomplete MLX SDK: missing $mlx_root/$file" >&2; exit 1
    fi
done
if ! grep -q 'set(PACKAGE_VERSION "0.32.2")' "$mlx_root/share/cmake/MLX/MLXConfigVersion.cmake"; then
    echo 'This release pins MLX 0.32.2; provide its matching SDK.' >&2; exit 1
fi
# Build a static C API shim against the selected C++ runtime.
fetch_source mlx-c "$revision"
cmake -S "$work/mlx-c" -B "$work/build" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DMLX_C_USE_SYSTEM_MLX=ON \
    -DMLX_C_BUILD_EXAMPLES=OFF -DMLX_DIR="$mlx_root/share/cmake/MLX" \
    -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=26.2 -DCMAKE_INSTALL_PREFIX="$sdk"
cmake --build "$work/build" --parallel "$jobs"
cmake --install "$work/build"
mkdir -p "$output/lib" "$output/licenses"
output=$(cd "$output" && pwd)
export CGO_LDFLAGS_ALLOW='-Wl,-rpath,@executable_path/lib'
# Use temporary aliases so cgo's linker flags also work when the SDK or checkout
# path contains spaces. The aliases are build-time search paths, never rpaths.
link_dir=$(mktemp -d /tmp/tinyoai-link.XXXXXX)
trap 'rm -rf "$link_dir"' EXIT
ln -s "$sdk" "$link_dir/sdk"
ln -s "$mlx_root" "$link_dir/runtime"
export CGO_CFLAGS="-I$link_dir/sdk/include -mmacosx-version-min=26.2"
export CGO_LDFLAGS="-L$link_dir/sdk/lib -L$link_dir/runtime/lib -mmacosx-version-min=26.2 -Wl,-rpath,@executable_path/lib"
(cd "$repo" && go build -tags mlx -trimpath -ldflags='-s -w' -o "$output/tinyoai" ./cmd/tinyoai)
cp "$mlx_root/lib/libmlx.dylib" "$output/lib/"
cp "$mlx_root/lib/libjaccl.dylib" "$output/lib/"
cp "$mlx_root/lib/mlx.metallib" "$output/lib/"
# Resolve MLX's own native dependency relative to the bundled library.
if ! otool -l "$output/lib/libmlx.dylib" | grep 'path @loader_path (offset' >/dev/null; then
    install_name_tool -add_rpath @loader_path "$output/lib/libmlx.dylib"
fi
# Changing load paths invalidates signatures. Ad-hoc signing makes the local
# bundle runnable; Developer ID signing and notarization are separate steps.
for binary in "$output/lib/libjaccl.dylib" "$output/lib/libmlx.dylib" "$output/tinyoai"; do
    codesign --force --sign - "$binary"
done
# Carry notices with the binaries and record the exact toolchain and file hashes.
cp "$repo/LICENSE" "$output/licenses/tinyoai.txt"
cp "$repo/docs/MLX-LICENSE.txt" "$output/licenses/MLX-LM.txt"
cp "$work/mlx-c/LICENSE" "$output/licenses/MLX-C.txt"
# Use SDK notices when available; cache downloads for externally supplied SDKs.
copy_license() {
    local name=$1 url=$2 cached="$work/licenses/$1"
    if [[ -f "$mlx_root/share/licenses/$name" ]]; then
        cp "$mlx_root/share/licenses/$name" "$output/licenses/$name"
        return
    fi
    if [[ ! -s "$cached" ]]; then
        mkdir -p "$work/licenses"
        curl --fail --silent --show-error --location --retry 3 "$url" -o "$cached.tmp"
        mv "$cached.tmp" "$cached"
    fi
    cp "$cached" "$output/licenses/$name"
}
copy_license MLX.txt https://raw.githubusercontent.com/ml-explore/mlx/v0.32.2/LICENSE
# JACCL is part of the MLX source tree and shares its license.
cp "$output/licenses/MLX.txt" "$output/licenses/JACCL.txt"
copy_license nlohmann-json.txt https://raw.githubusercontent.com/nlohmann/json/v3.11.3/LICENSE.MIT
copy_license fmt.txt https://raw.githubusercontent.com/fmtlib/fmt/12.1.0/LICENSE
cp "$mlx_root/include/metal_cpp/LICENSE.txt" "$output/licenses/metal-cpp.txt"
cp "$repo/docs/mlx-native.md" "$output/README.md"
{
    echo "MLX runtime: 0.32.2"
    echo "MLX SDK: $mlx_root"
    echo "MLX C revision: $revision"
    echo "Build time (UTC): $(date -u +%FT%TZ)"
    go version
    echo "GOEXPERIMENT=$GOEXPERIMENT"
} > "$output/BUILD.txt"
(cd "$output" && find . -type f ! -name SHA256SUMS -exec shasum -a 256 {} \; > SHA256SUMS)
echo "Release ready: $output/tinyoai"
