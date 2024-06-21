responses = {
    # Caos do Regex
    r"Ambiguous plugin name `(\S+)' for files `(plugins\/[^\']+)' and `(plugins\/[^\']+)' in `plugins'":
    "Há múltiplos arquivos de plugins para o plugin '{0}': '{1}' e '{2}'.",
    r"Could not load 'plugins/([^']*)' in 'plugins'.*?Unknown/missing dependency plugins: \[([^\]]*)\]":
    "Não foi possível carregar o plugin `{0}` devido à dependência ausente `{1}`.",
    r"Unable to access jarfile ([^']*)":
    "Não foi possível acessar o arquivo jar `{0}` do servidor, verifique nos seus parâmetros de inicialização e as permissões",
    r"Current Java is ([^']*) but we require at least ([^']*)":
    "O seu servidor pede a versão Java `{1}`, porém está configurado para iniciar com a versão `{0}`",
    # Sem caos do Regex
    "The received string length is longer than maximum allowed":
    "Você, ou o servidor, contém um nickname, scoreboard, tag, ou similar, com um tamanho maior do que o permitido.",
}