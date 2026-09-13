import { Component } from 'react'

export default class ErrorBoundary extends Component {
  state = { error: null }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    console.error('UI crashed:', error, info?.componentStack)
  }

  render() {
    if (!this.state.error) return this.props.children
    return (
      <div className="flex h-full items-center justify-center bg-ink p-6">
        <div role="alert" className="max-w-md rounded-xl border border-ash-700 bg-ash-900 p-6 text-center">
          <h1 className="font-display text-2xl font-semibold text-parchment">Щось пішло не так</h1>
          <p className="mt-2 text-ash-400">
            Інтерфейс натрапив на непередбачену помилку. Історія кампанії зберігається на сервері — перезавантаження
            сторінки нічого не зітре.
          </p>
          <p className="mt-3 break-words font-mono text-xs text-ash-600">{String(this.state.error?.message ?? this.state.error)}</p>
          <button
            type="button"
            onClick={() => window.location.reload()}
            className="mt-5 rounded border border-ember/60 bg-ember/20 px-4 py-2 text-parchment hover:bg-ember/35"
          >
            Перезавантажити
          </button>
        </div>
      </div>
    )
  }
}
