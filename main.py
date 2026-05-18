import click
import yaml
import json
from pathlib import Path
from src.pipeline import Pipeline, configs, preprocess_configs

@click.group()
def cli():
    """Pipeline command line interface for processing PDF reports and questions."""
    pass

@cli.command()
def download_models():
    """Download required docling models."""
    click.echo("Downloading docling models...")
    Pipeline.download_docling_models()

@cli.command()
@click.option('--parallel/--sequential', default=True, help='Run parsing in parallel or sequential mode')
@click.option('--chunk-size', default=2, help='Number of PDFs to process in each worker')
@click.option('--max-workers', default=10, help='Number of parallel worker processes')
def parse_pdfs(parallel, chunk_size, max_workers):
    """Parse PDF reports with optional parallel processing."""
    root_path = Path.cwd()
    pipeline = Pipeline(root_path)
    
    click.echo(f"Parsing PDFs (parallel={parallel}, chunk_size={chunk_size}, max_workers={max_workers})")
    pipeline.parse_pdf_documents(parallel=parallel, chunk_size=chunk_size, max_workers=max_workers)



@cli.command()
@click.option('--urls-file', default='urls.json', help='URLs file (.json/.yaml/.yml). Supports ["https://..."] or [{"url": "..."}]')
@click.option('--crawl-delay', default=0.5, help='Delay between URL requests in seconds')
def parse_urls(urls_file, crawl_delay):
    """Parse URL pages and save them in pipeline-compatible JSON format."""
    root_path = Path.cwd()
    pipeline = Pipeline(root_path)

    urls_path = root_path / urls_file
    if not urls_path.exists():
        raise click.ClickException(f"URLs file not found: {urls_path}")

    suffix = urls_path.suffix.lower()
    with urls_path.open('r', encoding='utf-8') as file:
        if suffix == '.json':
            urls = json.load(file)
        elif suffix in {'.yaml', '.yml'}:
            urls = yaml.safe_load(file)
        else:
            raise click.ClickException(f"Unsupported URLs file format: {suffix}. Use .json, .yaml, or .yml")

    if not isinstance(urls, list):
        raise click.ClickException("URLs file must contain a list of URLs or URL objects")

    click.echo(f"Parsing URLs from {urls_path}...")
    pipeline.parse_url_documents(urls=urls, crawl_delay=crawl_delay)


@cli.command()
@click.option('--max-workers', default=10, help='Number of workers for table serialization')
def serialize_tables(max_workers):
    """Serialize tables in parsed reports using parallel threading."""
    root_path = Path.cwd()
    pipeline = Pipeline(root_path)
    
    click.echo(f"Serializing tables (max_workers={max_workers})...")
    pipeline.serialize_tables(max_workers=max_workers)

@cli.command()
@click.option('--config', type=click.Choice(['ser_tab', 'no_ser_tab']), default='no_ser_tab', help='Configuration preset to use')
def process_documents(config):
    """Process parsed source documents through the pipeline stages."""
    root_path = Path.cwd()
    run_config = preprocess_configs[config]
    pipeline = Pipeline(root_path, run_config=run_config)
    
    click.echo(f"Processing parsed documents (config={config})...")
    pipeline.process_parsed_documents()

@cli.command()
@click.option('--config', type=click.Choice(['base', 'pdr', 'max', 'max_no_ser_tab', 'max_nst_o3m', 'max_st_o3m', 'ibm_llama70b', 'ibm_llama8b', 'gemini_thinking']), default='base', help='Configuration preset to use')
def process_questions(config):
    """Process questions using the pipeline."""
    root_path = Path.cwd()
    run_config = configs[config]
    pipeline = Pipeline(root_path, run_config=run_config)
    
    click.echo(f"Processing questions (config={config})...")
    pipeline.process_questions()

if __name__ == '__main__':
    cli()