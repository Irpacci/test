#include <cstdint>
#include <algorithm>
#include <array>
#include <immintrin.h> // Для AVX-512 инструкций

#pragma GCC optimize("O3", "unroll-loops", "omit-frame-pointer", "inline")
#pragma GCC target("avx512f", "avx512bw", "avx512dq", "avx512cd")

struct Update {
    uint32_t price;
    uint32_t quantity;
};

using UpdateRow = std::vector<Update>;
struct UpdateRowWithShares {
    UpdateRow row;
    uint32_t shares;
};
using Updates = std::vector<UpdateRowWithShares>;

uint64_t Solve(const Updates& updates) {
    uint64_t result = 0;

    // Массивы для хранения текущего состояния книги ордеров
    alignas(64) std::array<Update, 401> book1{};
    alignas(64) std::array<Update, 401> book2{};
    Update* currentBook = book1.data();
    Update* nextBook = book2.data();
    int currentSize = 0;

    for (const auto& [update, shares] : updates) {
        // Слияние текущего состояния с обновлением
        int i = 0, j = 0, k = 0;
        const int updateSize = update.size();

        while (i < currentSize && j < updateSize) {
            if (currentBook[i].price < update[j].price) {
                nextBook[k++] = currentBook[i++];
            } else if (currentBook[i].price > update[j].price) {
                nextBook[k++] = update[j++];
            } else {
                if (update[j].quantity > 0) {
                    nextBook[k++] = update[j];
                }
                ++i;
                ++j;
            }
        }

        while (i < currentSize) {
            nextBook[k++] = currentBook[i++];
        }

        while (j < updateSize) {
            nextBook[k++] = update[j++];
        }

        currentSize = k;

        // Вычисление стоимости покупки S акций с использованием AVX-512
        uint64_t cost = 0;
        uint32_t sharesRemaining = shares;

        // Развернутый цикл с использованием AVX-512
        int idx = 0;
        __m512i zero = _mm512_setzero_epi32();
        __m512i sharesVec = _mm512_set1_epi32(sharesRemaining);

        for (; idx <= currentSize - 16; idx += 16) {
            __m512i prices = _mm512_loadu_epi32(reinterpret_cast<const int*>(&nextBook[idx].price));
            __m512i quantities = _mm512_loadu_epi32(reinterpret_cast<const int*>(&nextBook[idx].quantity));

            __m512i sharesToBuy = _mm512_min_epu32(quantities, sharesVec);
            __m512i costs = _mm512_mullo_epi32(sharesToBuy, prices);

            // Суммируем результаты
            int32_t temp[16];
            _mm512_storeu_epi32(temp, costs);
            for (int t = 0; t < 16; ++t) {
                cost += temp[t];
            }

            // Обновляем оставшееся количество акций
            __m512i remaining = _mm512_sub_epi32(sharesVec, sharesToBuy);
            _mm512_storeu_epi32(temp, remaining);
            sharesRemaining = temp[0]; // Берем первый элемент как остаток
            if (sharesRemaining == 0) break;
        }

        // Обработка оставшихся элементов
        for (; idx < currentSize && sharesRemaining > 0; ++idx) {
            uint32_t availableShares = nextBook[idx].quantity;
            uint32_t sharesToBuy = std::min(availableShares, sharesRemaining);
            cost += static_cast<uint64_t>(sharesToBuy) * nextBook[idx].price;
            sharesRemaining -= sharesToBuy;
        }

        result ^= cost;

        // Переключение указателей на текущую и следующую книги ордеров
        std::swap(currentBook, nextBook);
    }

    return result;
}
